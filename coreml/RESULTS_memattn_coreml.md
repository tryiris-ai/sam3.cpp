# EdgeTAM memory_attention -> CoreML / ANE — results (2026-06-16)

Frontier export of EdgeTAM's MEMORY ATTENTION module (the ~185 ms/frame ggml/Metal
bottleneck) to CoreML at fixed capacity, mirroring the encoder precedent.

## Outcome: CONVERTED + RUNS. Parity ~1.0. But ANE is NOT the win for this module.

Artifacts (all in this dir):
- `convert_memattn_coreml.py` — wrap → trace → freeze → run_frozen_optimizations → ct.convert
- `bench_memattn_coreml.py`   — per-compute-unit benchmark + parity vs PyTorch
- `coreml_models/edgetam_memory_attention.mlpackage` — the converted FP16 model (8.6 MB)

## Memory-attention forward signature (confirmed by source + live model)
`MemoryAttention.forward(curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens, num_spatial_mem)`
- `curr`       [N_img, B, D]  self-attn (current-frame) tokens
- `memory`     [N_mem, B, mem_dim] cross-attn (memory) tokens
- `curr_pos`   [N_img, B, D]  pos enc for curr
- `memory_pos` [N_mem, B, mem_dim] pos enc for memory
- `num_obj_ptr_tokens` int (rope-excluded tail of memory)
- `num_spatial_mem` int (= rope_k_repeat: number of spatial-memory frames)

## Fixed shapes used (EdgeTAM steady-state, 1024² input)
- B=1, D=256, mem_dim=64, num_heads=1
- N_img = 64×64 = **4096** current-frame tokens
- memory = N_spatial(7)×512 + obj_ptr(64) = **3648** tokens
  - per spatial frame: 512 = 256 (1d perceiver latents, NO rope) + 256 (2d window latents, ROPE)
  - 7 spatial frames (num_maskmem cap) + 16 obj-ptrs × (C//mem_dim=4) = 64 obj-ptr tokens (rope-excluded)

## Walls hit and how they were cleared
1. **Complex RoPE (the anticipated wall).** `sam2/modeling/position_encoding.py`
   `apply_rotary_enc` / `apply_rotary_enc_v2` use `torch.view_as_complex` / `torch.polar` /
   `torch.view_as_real` (lines 200/207, 246/264) — coremltools rejects these.
   **Fix:** monkey-patched both `RoPEAttention.forward` and `RoPEAttentionv2.forward` with a
   REAL-VALUED RoPE: precompute cos/sin for the exact fixed grids, apply the rotation as
   `(a+ib)(cos+isin) = (a cos − b sin) + i(a sin + b cos)` via interleaved even/odd slices.
   Verified numerically equal to the complex path to ~5e-7 (all 3 paths: self-attn, cross-q,
   cross-k with 7× repeat + rope/no-rope split + obj-ptr exclusion).

2. **`"only 0-dimensional arrays can be converted to Python scalars"` (the encoder-precedent
   error) — but freeze did NOT fold it here.** Source: `transformer.py:250` `_separate_heads`
   does `b,n,c = x.shape; x.reshape(b, n, num_heads, c // num_heads)`; the `c // num_heads`
   becomes `aten::size → prim::NumToTensor → aten::floor_divide → aten::Int`, which the
   coremltools `_int` op handler cannot const-fold even after `torch.jit.freeze`. Also
   `RoPEAttention.forward` has `w=h=math.sqrt(q.shape[-2])` (dynamic recompute branch).
   **Fix:** replaced `_separate_heads`/`_recombine_heads` and the RoPE reshapes with
   CONSTANT-SHAPE reshapes (literal ints for B/heads/tokens/head_dim) — no `.shape` arithmetic
   leaks into the graph. This dropped the op count 519 → 296 and removed every dynamic op.
   (Note: `torch.jit.freeze` was still applied per the precedent; it folds the buffer/const
   ops, but the dynamic shape ops had to be removed at the source by constant reshapes.)

3. **CoreML rank ≤ 5.** The per-frame k-rope layout `[1,nh,n_spatial,rope_tok,half,2]` is rank 6.
   **Fix:** pre-tile k cos/sin across the 7 frames and rotate the rope sub-block as a single
   rank-4 `[1,nh,n_spatial*rope_tok,hd]` tensor.

After these, the PyTorch frontend converts 100% (296/296 MIL ops) and the backend emits the
mlprogram cleanly. No remaining blockers.

## Parity (CoreML output vs original COMPLEX-RoPE PyTorch module, identical inputs)
- patched real-RoPE vs original complex (PyTorch/PyTorch): cosine = **1.000000**, maxdiff 1.3e-6
- CoreML vs PyTorch reference: cosine **0.999991–1.000000** across all compute units (FP16)

## Benchmark (5 warmup + 20 timed predict(), fixed-shape inputs; M4 Pro)
Per-call dispatch overhead measured at ~0.03 ms (negligible) — the numbers below are compute.

| compute_unit  | mean ms | min ms | cosine vs torch |
|---------------|--------:|-------:|----------------:|
| CPU_ONLY      |  23.5   | 22.6   | 0.999991 |
| CPU_AND_GPU   | **17.1**| 16.8   | 1.000000 |
| CPU_AND_NE    |  26.4   | 26.1   | 0.999996 |
| ALL           |  32.1   | 31.5   | 0.999996 |

Reference: ggml/Metal mem_attn = **185 ms/frame**.

## HONEST headline
- **Best CoreML latency = 17 ms on the GPU = ~10.9× vs 185 ms ggml/Metal.** Real, parity-clean.
- **The ANE is NOT the win for memory attention** (26 ms, slower than GPU, ~CPU-tie). This is
  the OPPOSITE of the encoder (12 ms ANE / 10× over ggml). The static CoreML compute plan
  reports 100% of ops "preferred" on `MLNeuralEngineComputeDevice`, but wall-clock proves the
  ANE does not actually accelerate this graph: a single-head attention over a 4096×3648 score
  matrix (big matmul + softmax) is GPU-favorable, whereas the convolutional RepViT encoder is
  ANE-favorable. `powermetrics` ANE-power confirmation needs sudo (not run).
- So for the RFD: route the memory-attention bottleneck to **CoreML-GPU (17 ms)**, not the ANE.
  That still removes the ~185 ms ggml mem_attn stall (the warmed-frame compute bound from the
  sam3.cpp eval). The encoder stays on ANE; mem-attn goes to GPU.

## Reproduce
```
~/.cache/edgetam-coreml-venv/bin/python convert_memattn_coreml.py   # builds the mlpackage
~/.cache/edgetam-coreml-venv/bin/python bench_memattn_coreml.py     # benchmarks + parity
```
