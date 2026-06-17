# EdgeTAM MEMORY ATTENTION -> CoreML (fixed-capacity), frontier export.
#
# Approach: the SAM2 memory attention uses RoPE implemented with torch.view_as_complex /
# torch.polar / torch.view_as_real (sam2/modeling/position_encoding.py), which coremltools
# cannot convert, AND a dynamic `math.sqrt(q.shape[-2])` recompute branch that produces the
# "only 0-dimensional arrays can be converted to Python scalars" error even after freeze.
#
# Fix (per the RFD's fixed-capacity design + Egor Dmitriev's real-valued-RoPE precedent):
# monkey-patch RoPEAttention.forward and RoPEAttentionv2.forward with REAL-VALUED RoPE that
# (a) uses precomputed cos/sin buffers for the EXACT fixed grids (no math.sqrt, no recompute),
# (b) replaces complex multiply with the real interleaved rotation
#     (a+ib)(cos+isin) = (a cos - b sin) + i(a sin + b cos).
# Real-valued RoPE verified to match the complex path to ~5e-7 (see /tmp/verify_rope_real*.py).
#
# Then: trace -> torch.jit.freeze -> run_frozen_optimizations -> ct.convert (mlprogram, iOS17).

import os, sys, warnings; warnings.filterwarnings("ignore")
import torch, numpy as np
import torch.nn.functional as F
import coremltools as ct
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval")
OUT = f"{GE}/coreml_models/edgetam_memory_attention.mlpackage"

# ----- Fixed capacity (EdgeTAM steady state) -----
B, D, MEM_DIM = 1, 256, 64
N_IMG = 64 * 64                  # 4096 current-frame tokens (1024/16=64)
N_SPATIAL = 7                    # num_maskmem cap -> 7 spatial memory frames
TOK_PER_FRAME = 512              # spatial perceiver: 256 (1d, no-rope) + 256 (2d, rope)
N_OBJ_PTR = 16 * 4               # 16 obj ptrs x (C//mem_dim=4) = 64 obj-ptr tokens (rope-excluded)
N_MEM = N_SPATIAL * TOK_PER_FRAME + N_OBJ_PTR   # 3648


def build_model():
    GlobalHydra.instance().clear()
    cfgp = f"{GE}/EdgeTAM/sam2/configs/edgetam.yaml"
    with initialize_config_dir(config_dir=os.path.dirname(cfgp), version_base=None):
        cfg = compose(config_name="edgetam"); OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
    from sam2.build_sam import _load_checkpoint
    _load_checkpoint(model, f"{GE}/edgetam_ckpt/edgetam.pt"); model.eval()
    return model


def _rope_real(x, cos, sin, n_heads, n_tokens, head_dim):
    # x:[1,H,N,C]; cos/sin:[N,C/2]. Real interleaved rotation matching view_as_complex pairing
    # (even idx = real a, odd idx = imag b). All dims passed as Python-int constants so no
    # dynamic shape arithmetic (Cx//2) enters the traced graph.
    half = head_dim // 2
    xr = x.reshape(1, n_heads, n_tokens, half, 2)
    a = xr[..., 0]; b = xr[..., 1]
    c = cos.view(1, 1, n_tokens, half); s = sin.view(1, 1, n_tokens, half)
    out_a = a * c - b * s
    out_b = a * s + b * c
    return torch.stack((out_a, out_b), dim=-1).reshape(1, n_heads, n_tokens, head_dim)


def _sep_heads_const(x, n_heads, n_tokens, head_dim):
    # Constant-shape reshape; avoids aten::size/NumToTensor/floor_divide/Int that coremltools
    # cannot const-fold (the original _separate_heads does `b,n,c = x.shape; c//num_heads`).
    x = x.reshape(1, n_tokens, n_heads, head_dim)
    return x.transpose(1, 2)  # [1, n_heads, n_tokens, head_dim]


def _recomb_heads_const(x, n_heads, n_tokens, head_dim):
    x = x.transpose(1, 2)
    return x.reshape(1, n_tokens, n_heads * head_dim)


def patch_attentions(ma):
    """Replace the two RoPE attention forwards with fixed-shape, complex-free versions.

    Buffers are precomputed for the EXACT runtime grids so there is no dynamic recompute:
      - self_attn (RoPEAttention): q,k both length N_IMG=4096 -> grid 64x64.
      - cross_attn (RoPEAttentionv2): q length 4096 (64x64); k rope tokens 256 (16x16),
        repeated N_SPATIAL times, last N_OBJ_PTR tokens excluded from rope.

    Also replaces _separate_heads/_recombine_heads with constant-shape reshapes so no
    dynamic shape arithmetic (aten::Int on prim::NumToTensor) leaks into the graph.
    """
    from sam2.modeling.position_encoding import compute_axial_cis

    for layer in ma.layers:
        sa = layer.self_attn          # RoPEAttention
        ca = layer.cross_attn_image   # RoPEAttentionv2

        # ----- self-attn: real cos/sin for the actual 4096-token (64x64) grid -----
        head_dim_sa = sa.internal_dim // sa.num_heads
        cis_sa = compute_axial_cis(dim=head_dim_sa, end_x=64, end_y=64,
                                   theta=10000.0)  # [4096, head_dim/2] complex
        sa_cos = cis_sa.real.contiguous(); sa_sin = cis_sa.imag.contiguous()

        def make_sa_forward(sa, sa_cos, sa_sin, nh, ntok, hd):
            def forward(q, k, v, num_k_exclude_rope: int = 0):
                q = sa.q_proj(q); k = sa.k_proj(k); v = sa.v_proj(v)
                q = _sep_heads_const(q, nh, ntok, hd)
                k = _sep_heads_const(k, nh, ntok, hd)
                v = _sep_heads_const(v, nh, ntok, hd)
                # self-attn here always has q,k == full image grid, num_k_exclude_rope==0
                q = _rope_real(q, sa_cos, sa_sin, nh, ntok, hd)
                k = _rope_real(k, sa_cos, sa_sin, nh, ntok, hd)
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                out = _recomb_heads_const(out, nh, ntok, hd)
                return sa.out_proj(out)
            return forward
        sa.forward = make_sa_forward(sa, sa_cos, sa_sin, sa.num_heads, N_IMG, head_dim_sa)

        # ----- cross-attn (v2): q grid 64x64=4096; k grid 16x16=256 repeated -----
        head_dim_ca = ca.internal_dim // ca.num_heads
        cis_q = compute_axial_cis(dim=head_dim_ca, end_x=64, end_y=64, theta=10000.0)  # [4096, hd/2]
        cis_k = compute_axial_cis(dim=head_dim_ca, end_x=16, end_y=16, theta=10000.0)  # [256,  hd/2]
        q_cos = cis_q.real.contiguous(); q_sin = cis_q.imag.contiguous()
        ROPE_TOK = cis_k.shape[0]          # 256 rope tokens per frame (the 2d window latents)
        # Pre-tile k cos/sin across the N_SPATIAL frames so the rope sub-block can be processed
        # as a single rank-4 [1, nh, N_SPATIAL*ROPE_TOK, hd] tensor (CoreML max rank is 5; a
        # per-frame [.,.,n_spatial,rope_tok,half,2] layout would be rank 6 and is rejected).
        k_cos = cis_k.real.repeat(N_SPATIAL, 1).contiguous()   # [N_SPATIAL*256, hd/2]
        k_sin = cis_k.imag.repeat(N_SPATIAL, 1).contiguous()

        def make_ca_forward(ca, q_cos, q_sin, k_cos, k_sin, rope_tok,
                            n_spatial, tok_per_frame, n_obj, nh, n_q, n_kv, hd):
            def forward(q, k, v, num_k_exclude_rope: int = 0, rope_k_repeat: int = -1):
                q = ca.q_proj(q); k = ca.k_proj(k); v = ca.v_proj(v)
                q = _sep_heads_const(q, nh, n_q, hd)
                k = _sep_heads_const(k, nh, n_kv, hd)
                v = _sep_heads_const(v, nh, n_kv, hd)
                # q: full image grid, repeat_freqs=1 -> straight rotation
                q = _rope_real(q, q_cos, q_sin, nh, n_q, hd)
                # k: first (n_spatial*tok_per_frame) tokens are spatial memory; the last n_obj
                # are object pointers (rope-excluded). Within each frame of tok_per_frame=512,
                # the first (512-256)=256 are no-rope (1d latents), the last 256 are rope (2d).
                # All dims are Python-int constants (no k.shape) so nothing dynamic enters the graph.
                half = hd // 2
                n_rope_block = n_spatial * tok_per_frame              # 3584
                k_spatial = k[:, :, :n_rope_block, :]
                k_obj = k[:, :, n_rope_block:, :]                     # [.,.,n_obj,.] untouched
                no_rope = tok_per_frame - rope_tok                   # 256
                ksp = k_spatial.reshape(1, nh, n_spatial, tok_per_frame, hd)  # rank 5
                k_no = ksp[:, :, :, :no_rope, :]                     # 1d latents, no rope (rank 5)
                k_ro = ksp[:, :, :, no_rope:, :]                     # 2d latents, rope (rank 5)
                # Flatten frames into a single token axis so rotation stays rank<=5:
                #   [1, nh, n_spatial, rope_tok, hd] -> [1, nh, n_spatial*rope_tok, hd] (rank 4)
                n_kr = n_spatial * rope_tok
                k_ro = k_ro.reshape(1, nh, n_kr, hd)
                kr = k_ro.reshape(1, nh, n_kr, half, 2)             # rank 5 (OK)
                a = kr[..., 0]; b = kr[..., 1]
                c = k_cos.view(1, 1, n_kr, half)                    # cos pre-tiled over frames
                s = k_sin.view(1, 1, n_kr, half)
                out_a = a * c - b * s
                out_b = a * s + b * c
                k_ro = torch.stack((out_a, out_b), dim=-1).reshape(1, nh, n_kr, hd)  # rank 4
                k_ro = k_ro.reshape(1, nh, n_spatial, rope_tok, hd)  # back to rank 5 to concat
                ksp = torch.cat((k_no, k_ro), dim=3).reshape(1, nh, n_rope_block, hd)
                k = torch.cat((ksp, k_obj), dim=2)
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                out = _recomb_heads_const(out, nh, n_q, hd)  # output has q-length tokens
                return ca.out_proj(out)
            return forward
        ca.forward = make_ca_forward(ca, q_cos, q_sin, k_cos, k_sin, ROPE_TOK,
                                     N_SPATIAL, TOK_PER_FRAME, N_OBJ_PTR,
                                     ca.num_heads, N_IMG, N_MEM, head_dim_ca)


class Wrapper(torch.nn.Module):
    def __init__(self, ma, n_obj, n_spatial):
        super().__init__()
        self.ma = ma; self.n_obj = n_obj; self.n_spatial = n_spatial

    def forward(self, curr, memory, curr_pos, memory_pos):
        return self.ma(curr=curr, memory=memory, curr_pos=curr_pos, memory_pos=memory_pos,
                       num_obj_ptr_tokens=self.n_obj, num_spatial_mem=self.n_spatial)


def main():
    model = build_model()
    ma = model.memory_attention
    patch_attentions(ma)
    w = Wrapper(ma, N_OBJ_PTR, N_SPATIAL).eval()

    d = np.load("/tmp/memattn_inputs.npz")
    curr = torch.from_numpy(d["curr"]); curr_pos = torch.from_numpy(d["curr_pos"])
    memory = torch.from_numpy(d["memory"]); memory_pos = torch.from_numpy(d["memory_pos"])
    ex = (curr, memory, curr_pos, memory_pos)

    # Parity check: patched (real RoPE) vs original complex reference output.
    with torch.no_grad():
        out_patched = w(*ex)
    ref = np.load("/tmp/memattn_ref_out.npy")
    cos = np.dot(out_patched.numpy().ravel(), ref.ravel()) / (
        np.linalg.norm(out_patched.numpy().ravel()) * np.linalg.norm(ref.ravel()))
    print(f"[parity] patched-real-RoPE vs original-complex cosine = {cos:.6f}")
    print(f"[parity] max abs diff = {np.abs(out_patched.numpy()-ref).max():.3e}")

    print("[1] tracing patched module ...")
    traced = torch.jit.trace(w, ex, check_trace=False)
    print("    trace OK")

    print("[2] freeze + run_frozen_optimizations ...")
    frozen = torch.jit.freeze(traced)
    torch.jit.run_frozen_optimizations(frozen)
    print("    freeze OK")

    print("[3] ct.convert (mlprogram, iOS17, ALL) ...")
    mlmodel = ct.convert(
        frozen,
        inputs=[
            ct.TensorType(name="curr",       shape=curr.shape,       dtype=np.float32),
            ct.TensorType(name="memory",     shape=memory.shape,     dtype=np.float32),
            ct.TensorType(name="curr_pos",   shape=curr_pos.shape,   dtype=np.float32),
            ct.TensorType(name="memory_pos", shape=memory_pos.shape, dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mlmodel.save(OUT)
    print(f"    CONVERT OK -> {OUT}")


if __name__ == "__main__":
    main()
