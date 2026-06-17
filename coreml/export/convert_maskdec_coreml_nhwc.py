# EdgeTAM SAM MASK DECODER -> CoreML, CHANNELS-LAST (NHWC) input variant.
#
# This is convert_maskdec_coreml.py with ONE change: the wrapper's spatial inputs
# are NHWC (channels-innermost) instead of NCHW. The C++ ggml tracker stores its
# feature buffers channels-innermost, so an NHWC-input CoreML model lets it hand
# its buffers to CoreML with a plain memcpy -- no per-frame transpose. We permute
# NHWC->NCHW at the very start of forward(), then run the EXACT same body
# (conv_s0/conv_s1 + md.predict_masks + the same 4 outputs), with the EXACT same
# monkey-patches (_separate_heads / predict_masks constant-shape rewrites).
#
# OUTPUTS ARE UNCHANGED (still NCHW-shaped): masks [1,4,256,256], iou_pred [1,4],
# obj_score [1,1], mask_tokens [1,4,256]. We do NOT permute outputs -- the C++
# side already knows how to consume the decoder's native output layout.
#
# Everything below the input-permute is copied verbatim from the NCHW converter so
# the two stay in lock-step. See convert_maskdec_coreml.py for the full rationale
# on the freeze + run_frozen_optimizations trick and the two monkey-patches.

import os, warnings; warnings.filterwarnings("ignore")
import torch, coremltools as ct, numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval")
OUT = f"{GE}/coreml_models/edgetam_mask_decoder_nhwc.mlpackage"

# ----- Fixed input shapes -- CHANNELS-LAST (NHWC) for the 5 spatial inputs -----
# Same tensors as the NCHW converter, just with the channel axis moved last so the
# ggml tracker (channels-innermost buffers) can memcpy straight in:
#   image_embeddings/image_pe/dense: [1,64,64,256]  (was [1,256,64,64])
#   feat_s0:  [1,256,256,256]  (H=256,W=256,C=256 -- square, so the tuple looks the
#             same as NCHW, but it is now interpreted N,H,W,C)
#   feat_s1:  [1,128,128,256]  (was [1,256,128,128])
#   sparse:   [1,1,256]        (unchanged -- not a spatial map)
SHAPES = {
    "image_embeddings": (1, 64, 64, 256),
    "image_pe":         (1, 64, 64, 256),
    "sparse":           (1, 1, 256),
    "dense":            (1, 64, 64, 256),
    "feat_s0":          (1, 256, 256, 256),
    "feat_s1":          (1, 128, 128, 256),
}


def build_model():
    GlobalHydra.instance().clear()
    cfgp = f"{GE}/EdgeTAM/sam2/configs/edgetam.yaml"
    with initialize_config_dir(config_dir=os.path.dirname(cfgp), version_base=None):
        cfg = compose(config_name="edgetam"); OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
    from sam2.build_sam import _load_checkpoint
    _load_checkpoint(model, f"{GE}/edgetam_ckpt/edgetam.pt"); model.eval()
    return model


def patch_decoder_attention(md):
    """Constant-shape _separate_heads/_recombine_heads on every Attention submodule.

    Root blocker: Attention._separate_heads does `b,n,c = x.shape; reshape(b,n,
    num_heads, c//num_heads)`. The `c//num_heads` integer divide becomes a traced
    aten::Int that coremltools cannot const-fold even after freeze, raising
    "only 0-dimensional arrays can be converted to Python scalars". head_dim is
    actually a compile-time constant (internal_dim // num_heads), so we capture it
    as a Python int and use -1 for the token axis -- killing the dynamic int.
    """
    from sam2.modeling.sam.transformer import Attention
    import types

    n_patched = 0
    for mod in md.modules():
        if isinstance(mod, Attention):
            nh = mod.num_heads
            hd = mod.internal_dim // mod.num_heads  # Python int constant

            def make_sep(nh, hd):
                def _separate_heads(self, x, num_heads):
                    # b is always 1 here; n stays symbolic via -1 so no aten::Int
                    # from c//num_heads leaks into the traced graph.
                    x = x.reshape(1, -1, nh, hd)
                    return x.transpose(1, 2)
                return _separate_heads

            def make_recomb(nh, hd):
                def _recombine_heads(self, x):
                    x = x.transpose(1, 2)
                    return x.reshape(1, -1, nh * hd)
                return _recombine_heads

            mod._separate_heads = types.MethodType(make_sep(nh, hd), mod)
            mod._recombine_heads = types.MethodType(make_recomb(nh, hd), mod)
            n_patched += 1
    print(f"[patch] constant-shape head split on {n_patched} Attention submodules")


def patch_predict_masks(md, grid_h, grid_w):
    """Constant-shape predict_masks: replace the two `b,c,h,w=x.shape; x.view(...)`
    reshapes whose dynamic ints survive freeze (the remaining 'only 0-dimensional
    arrays' error near the end of the graph, at the final mask reshape).

    This mirrors mask_decoder.MaskDecoder.predict_masks line-for-line, with the
    ONLY change being literal-int reshapes:
      - src reshape:   transformer_dim @ grid_h x grid_w   (256 @ 64x64)
      - mask reshape:  upscaled (transformer_dim//8) @ 4*grid_h x 4*grid_w (32 @ 256x256)
    All math/weights are untouched -> numerically identical (verified by the
    patch-parity check in main()).
    """
    import types

    D = md.transformer_dim                  # 256
    C_up = md.transformer_dim // 8          # 32 (output_upscaling final channels)
    H4, W4 = grid_h * 4, grid_w * 4         # 256, 256 (high-res mask grid)
    n_mask = md.num_mask_tokens             # 4

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings, repeat_image, high_res_features=None):
        # --- token concat (identical to source) ---
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [self.obj_score_token.weight, self.iou_token.weight,
                 self.mask_tokens.weight], dim=0)
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        # --- transformer ---
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # CONST reshape: [1, D, grid_h, grid_w] (was view(b,c,h,w) w/ dynamic dims)
        src = src.transpose(1, 2).reshape(1, D, grid_h, grid_w)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        # CONST reshape: upscaled -> [1, C_up, H4*W4]; masks -> [1, n_mask, H4, W4]
        masks = (hyper_in @ upscaled_embedding.reshape(1, C_up, H4 * W4)).reshape(
            1, n_mask, H4, W4)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    md.predict_masks = types.MethodType(predict_masks, md)
    print(f"[patch] constant-shape predict_masks reshapes "
          f"(src {D}@{grid_h}x{grid_w}, masks {n_mask}x{H4}x{W4})")


class FullDecoder(torch.nn.Module):
    """Matches ggml's sam3_build_sam_dec_graph I/O, with CHANNELS-LAST inputs.

    The ONLY difference from the NCHW FullDecoder: the 5 spatial inputs arrive NHWC
    (channels-innermost, as the ggml tracker stores them) and are permuted to NCHW
    at the very top of forward(). Everything after that line is identical -- the
    same conv_s0/conv_s1 down-projection and the same predict_masks call returning
    the raw, un-sliced 4-token output. Outputs are NOT permuted.
    """

    def __init__(self, md):
        super().__init__()
        self.md = md

    def forward(self, image_embeddings, image_pe, sparse, dense, feat_s0, feat_s1):
        # NHWC -> NCHW for the spatial inputs (channels-last in from the ggml tracker).
        # sparse [1,1,256] is not a spatial map -> left untouched.
        image_embeddings = image_embeddings.permute(0, 3, 1, 2).contiguous()
        image_pe         = image_pe.permute(0, 3, 1, 2).contiguous()
        dense            = dense.permute(0, 3, 1, 2).contiguous()
        feat_s0          = feat_s0.permute(0, 3, 1, 2).contiguous()
        feat_s1          = feat_s1.permute(0, 3, 1, 2).contiguous()

        # ----- identical to the NCHW converter from here down -----
        hr0 = self.md.conv_s0(feat_s0)   # 256 -> 32  (ggml dec.conv_s0)
        hr1 = self.md.conv_s1(feat_s1)   # 256 -> 64  (ggml dec.conv_s1)
        # predict_masks returns the un-sliced 4-token output that ggml emits.
        masks, iou_pred, mask_tokens, obj_score = self.md.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=[hr0, hr1],
        )
        # masks [1,4,256,256] | iou_pred [1,4] | obj_score [1,1] | mask_tokens [1,4,256]
        return masks, iou_pred, obj_score, mask_tokens


def make_inputs():
    torch.manual_seed(0)
    return {k: torch.randn(*s, dtype=torch.float32) for k, s in SHAPES.items()}


def main():
    model = build_model()
    md = model.sam_mask_decoder
    w = FullDecoder(md).eval()

    inp = make_inputs()
    order = ["image_embeddings", "image_pe", "sparse", "dense", "feat_s0", "feat_s1"]
    ex = tuple(inp[k] for k in order)
    ref_names = ["masks", "iou_pred", "obj_score", "mask_tokens"]

    # Capture the UNPATCHED PyTorch reference first (on the NHWC dummy inputs), so
    # parity is measured against the true original decoder fed the same NHWC tensors.
    # The /tmp refs feed the benchmark script's parity check on the EXACT same tensors.
    with torch.no_grad():
        ref_orig = [r.numpy().copy() for r in w(*ex)]

    # Patch the dynamic shape ints, then verify the patches are numerically
    # lossless vs the original before we trace/convert.
    patch_decoder_attention(md)  # kill the c//num_heads dynamic int (see header)
    patch_predict_masks(md, grid_h=64, grid_w=64)  # kill the final view() dynamic ints
    with torch.no_grad():
        ref = w(*ex)
    print("[patch-parity] patched vs original PyTorch (should be ~0 / 1.0):")
    for n, ro, r in zip(ref_names, ref_orig, ref):
        rn = r.numpy()
        cos = float(np.dot(ro.ravel(), rn.ravel()) /
                    (np.linalg.norm(ro.ravel()) * np.linalg.norm(rn.ravel())))
        print(f"       {n}: maxdiff={np.abs(ro - rn).max():.3e}  cosine={cos:.8f}")

    # NHWC-specific /tmp paths so the NHWC bench reads its own inputs/refs and does
    # not collide with the NCHW converter's /tmp/maskdec_*.npz.
    np.savez("/tmp/maskdec_nhwc_inputs.npz", **{k: v.numpy() for k, v in inp.items()})
    np.savez("/tmp/maskdec_nhwc_ref.npz", **{n: r for n, r in zip(ref_names, ref_orig)})
    print("[ref] PyTorch wrapper output shapes (NHWC inputs):")
    for n, r in zip(ref_names, ref):
        print(f"       {n}: {tuple(r.shape)}")

    print("[1] tracing FullDecoder (NHWC) ...")
    traced = torch.jit.trace(w, ex, check_trace=False)
    print("    trace OK")

    print("[2] freeze + run_frozen_optimizations ...")
    frozen = torch.jit.freeze(traced)
    torch.jit.run_frozen_optimizations(frozen)
    print("    freeze OK")

    print("[3] ct.convert (mlprogram, iOS17, ALL) ...")
    mlmodel = ct.convert(
        frozen,
        inputs=[ct.TensorType(name=k, shape=SHAPES[k], dtype=np.float32) for k in order],
        outputs=[ct.TensorType(name=n) for n in ref_names],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mlmodel.save(OUT)
    print(f"    CONVERT OK -> {OUT}")

    # Print the actual CoreML I/O spec so we know the real input/output names/shapes.
    print("[4] CoreML I/O spec:")
    spec = mlmodel.get_spec()
    for i in spec.description.input:
        st = i.type.multiArrayType
        print(f"       IN  {i.name}: shape={list(st.shape)}")
    for o in spec.description.output:
        st = o.type.multiArrayType
        print(f"       OUT {o.name}: shape={list(st.shape)}")


if __name__ == "__main__":
    main()
