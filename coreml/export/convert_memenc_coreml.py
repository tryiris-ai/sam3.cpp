# EdgeTAM MEMORY-ENCODE (stage 4) -> CoreML. The last un-exported stage.
#
# Stage 4 = memory_encoder (conv-only: MaskDownSampler + Fuser/CXBlocks + projs, exports
# clean) + spatial_perceiver (PerceiverResampler, compresses to 512 latents). The wall is
# PerceiverResampler.forward_2d: it does dynamic windowing — `B,C,H,W = x.shape` then
# `window_size = H // num_window` feeding `.view(...)` in window_partition. Those traced-int
# reshapes are the same "only 0-dimensional arrays can be converted to Python scalars" class
# that mem-attn/decoder hit. Memory features are a FIXED 64x64 grid (encoder neck2), so we
# patch forward_1d/forward_2d to constant shapes (H=W=64, num_window=16, window_size=4) — the
# exact technique convert_memattn used for _separate_heads. Then trace->freeze->ct.convert.
import os, math, warnings; warnings.filterwarnings("ignore")
import torch, numpy as np
import torch.nn as nn
import coremltools as ct
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval")
OUT = f"{GE}/coreml_models/edgetam_memory_encode.mlpackage"
H = W = 64            # memory feature grid (1024 / total_stride 16)
C_PIX = 256           # pix_feat channels (encoder hidden_dim)


def build_model():
    GlobalHydra.instance().clear()
    cfgp = f"{GE}/EdgeTAM/sam2/configs/edgetam.yaml"
    with initialize_config_dir(config_dir=os.path.dirname(cfgp), version_base=None):
        cfg = compose(config_name="edgetam"); OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
    from sam2.build_sam import _load_checkpoint
    _load_checkpoint(model, f"{GE}/edgetam_ckpt/edgetam.pt"); model.eval()
    return model


def patch_perceiver(perc):
    """Replace forward_1d/forward_2d with constant-shape versions (fixed 64x64 input).
    Mirrors convert_memattn's _sep_heads_const trick: no x.shape-derived ints reach .view()."""
    Cdim = perc.latents_2d.shape[-1] if perc.num_latents_2d > 0 else perc.latents.shape[-1]
    num_window = int(math.sqrt(perc.num_latents_2d))   # 16
    window_size = H // num_window                       # 4

    # Same wall convert_memattn hit: PerceiverAttention._separate_heads does
    # `b,n,c = x.shape; x.reshape(b,n,nh, c//nh)` -> the c//nh is a shape-derived int cast
    # (aten::Int) coremltools can't const-fold. Replace head split/merge with reshape(1,-1,..)
    # using only constants + the -1 inferred token dim (no x.shape reads at all).
    def _patch_heads(attn):
        nh = attn.heads
        inner = attn.to_q.out_features
        hd = inner // nh
        # keep b,n symbolic (the 2d path has batch=num_windows=256, not 1); only the channel
        # split c//nh -> constant hd, and the merge nh*cph -> constant inner, were the int casts.
        def _sep(x, num_heads, _nh=nh, _hd=hd):
            b, n, c = x.shape
            return x.reshape(b, n, _nh, _hd).transpose(1, 2)
        def _recomb(x, _inner=inner):
            b, h, n, cph = x.shape
            return x.transpose(1, 2).reshape(b, n, _inner)
        attn._separate_heads = _sep
        attn._recombine_heads = _recomb
    for layer in perc.layers:
        _patch_heads(layer.attn)
        if getattr(layer, "use_self_attn", False):
            _patch_heads(layer.self_attn)

    def forward_1d(x, pos):
        # mirrors the original exactly; only dynamic expand(x.shape[0]) -> expand(1)
        latents = perc.latents.unsqueeze(0).expand(1, -1, -1)        # B=1 literal
        x = x.permute(0, 2, 3, 1).flatten(1, 2)                      # (1, 4096, C)
        if not perc.pos_enc_at_key_value:
            _pos = None
        if pos is not None:
            _pos = pos.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            _pos = None
        for layer in perc.layers:
            latents = layer(latents, x, _pos)
        if pos is not None:
            pos = torch.zeros_like(latents)
        latents = perc.norm(latents)
        return latents, pos

    def forward_2d(x):
        latents_2d = perc.latents_2d.unsqueeze(0).expand(1, -1, -1).view(-1, 1, Cdim)
        x = x.permute(0, 2, 3, 1)                                    # (1, H, W, C)
        # constant-shape window_partition (B=1, H=W=64, window_size=4 all literal)
        x = x.view(1, num_window, window_size, num_window, window_size, Cdim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, Cdim)
        x = x.flatten(1, 2)                                          # (256 windows, 16, C)
        for layer in perc.layers:
            latents_2d = layer(latents_2d, x)
        latents_2d = latents_2d.view(1, num_window, num_window, Cdim).permute(0, 3, 1, 2)
        pos_2d = perc.position_encoding(latents_2d)
        pos_2d = pos_2d.permute(0, 2, 3, 1).flatten(1, 2)
        latents_2d = latents_2d.permute(0, 2, 3, 1).flatten(1, 2)
        latents_2d = perc.norm(latents_2d)
        return latents_2d, pos_2d

    perc.forward_1d = forward_1d
    perc.forward_2d = forward_2d


class MemEnc(nn.Module):
    """memory_encoder + spatial_perceiver, the per-frame memory-write compute (stage 4)."""
    def __init__(self, mem_enc, perceiver, sigmoid_scale=1.0, sigmoid_bias=0.0):
        super().__init__()
        self.mem_enc = mem_enc
        self.perceiver = perceiver
        self.sigmoid_scale = sigmoid_scale
        self.sigmoid_bias = sigmoid_bias

    def forward(self, pix_feat, mask_logits):
        # FIX (RFD 0011 U8): replicate _encode_new_memory's mask processing exactly —
        # sigmoid(mask) * sigmoid_scale_for_mem_enc + sigmoid_bias_for_mem_enc
        # (EdgeTAM: 20.0 / -10.0), THEN skip the encoder's own sigmoid. The prior
        # export used skip_mask_sigmoid=False (sigmoid only, no scale/bias), feeding the
        # encoder a [0,1] mask vs the tracker's [-10,10] — which broke the slot
        # (goldeneval 0.000 / proto 0.022). With scale+bias baked in, the model takes
        # RAW logits and matches the tracker (proto 1.000; C++ 4-stage 0.94 @ ~14fps).
        mask_for_mem = torch.sigmoid(mask_logits) * self.sigmoid_scale + self.sigmoid_bias
        out = self.mem_enc(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        feats = out["vision_features"]
        pos = out["vision_pos_enc"][0]
        feats, pos = self.perceiver(feats, pos)
        return feats, pos


def main():
    model = build_model()
    w = MemEnc(model.memory_encoder, model.spatial_perceiver,
               sigmoid_scale=float(model.sigmoid_scale_for_mem_enc),
               sigmoid_bias=float(model.sigmoid_bias_for_mem_enc)).eval()

    pix_feat = torch.randn(1, C_PIX, H, W)
    mask = torch.randn(1, 1, 1024, 1024)
    ex = (pix_feat, mask)

    # reference (unpatched dynamic perceiver) BEFORE patching
    with torch.no_grad():
        ref_f, ref_p = w(*ex)
    ref_f = ref_f.numpy(); ref_p = ref_p.numpy()

    patch_perceiver(model.spatial_perceiver)

    # The two PositionEmbeddingSine calls (memory_encoder 64x64, perceiver 16x16) read spatial
    # dims at runtime -> aten::Int that ct.convert can't const-fold (same wall convert_memattn
    # dodged by precomputing RoPE buffers). Grids are FIXED, so capture each PE output once and
    # inject it as a constant. Parity is preserved (constant == the real PE output at that size).
    with torch.no_grad():
        pe_mem = model.memory_encoder.position_encoding(torch.zeros(1, 1, H, W))
        pe_perc = model.spatial_perceiver.position_encoding(torch.zeros(1, 1, 16, 16))
    model.memory_encoder.position_encoding.forward = (lambda *a, _c=pe_mem, **k: _c)
    model.spatial_perceiver.position_encoding.forward = (lambda *a, _c=pe_perc, **k: _c)
    print(f"[pe] injected constant PEs: mem {tuple(pe_mem.shape)}  perc {tuple(pe_perc.shape)}")

    with torch.no_grad():
        pf, pp = w(*ex)
    pf = pf.numpy(); pp = pp.numpy()
    cosf = np.dot(pf.ravel(), ref_f.ravel()) / (np.linalg.norm(pf.ravel()) * np.linalg.norm(ref_f.ravel()))
    print(f"[parity] features cosine = {cosf:.6f}  max|d| = {np.abs(pf-ref_f).max():.3e}  shape {pf.shape}")
    print(f"[parity] pos shape {pp.shape}")

    print("[1] trace ..."); traced = torch.jit.trace(w, ex, check_trace=False); print("    ok")
    print("[2] freeze ..."); frozen = torch.jit.freeze(traced); torch.jit.run_frozen_optimizations(frozen); print("    ok")
    print("[3] ct.convert (mlprogram, iOS17) ...")
    mlmodel = ct.convert(
        frozen,
        inputs=[ct.TensorType(name="pix_feat", shape=pix_feat.shape, dtype=np.float32),
                ct.TensorType(name="mask_logits", shape=mask.shape, dtype=np.float32)],
        # Stable output names so the .mm/cgo bridge doesn't depend on coremltools
        # auto-naming (var_NNN), which changes per export. (feats, pos) order.
        outputs=[ct.TensorType(name="mem_feats"), ct.TensorType(name="mem_pos")],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mlmodel.save(OUT)
    print(f"    CONVERT OK -> {OUT}")


if __name__ == "__main__":
    main()
