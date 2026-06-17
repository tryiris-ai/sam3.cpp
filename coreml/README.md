# EdgeTAM on CoreML / Apple Neural Engine (RFD 0011)

The CoreML/ANE serving path for EdgeTAM — the route to real-time on Apple
silicon, which the ggml/Metal runtime cannot reach (ggml is ~10× off Apple's
kernels on the heavy stages). **Both bottleneck stages — the image encoder and
the memory attention — are wired into the tracker and working.**

## Measured on M4 Pro (verified, FP16, parity vs PyTorch)

| Stage | ggml/Metal | best CoreML | speedup | best device | parity |
|---|--:|--:|--:|---|--:|
| Image encoder (RepViT+FPN) | 122 ms | **11 ms** | ~11× | **ANE** | cosine 0.992 |
| Memory attention | ~105 ms | **17.8 ms** | ~6× | **GPU** | cosine 1.000 |

**Heterogeneous placement matters:** the conv-heavy encoder is fastest on the
ANE; the attention (matmul+softmax) is fastest on the GPU. The ANE is NOT
universally faster.

## Live hybrid — both legs (DONE)
- **Encoder** (ANE): replaces the ggml RepViT+FPN in `edgetam_encode_image`. The
  256-ch neck features (`neck(trunk(x))[0:3]`, matching ggml `neck_trk`) are
  exported **channels-last `[1,H,W,D]`**, byte-identical to ggml `[D,W,H]`, so the
  load is a `memcpy`.
- **Memory attention** (GPU): replaces the ggml `sam3_build_mem_attn_graph` in
  `sam3_propagate_single` at the fixed steady-state capacity (7 memory frames ×
  512 + 16 obj-ptrs × 4 = **3648 tokens**). ggml's `[feature,token]` CPU buffers
  are byte-identical to the model's `[token,1,feature]`, so `curr`/`memory`/
  positions hand straight over and the output injects as the decoder's
  `cond_spatial`. Early/transition frames (capacity ≠ 3648) fall back to ggml.

**Measured (goldeneval, dancetrack, seed-once, tf@0.5):**

| | fps | dt0004 | dt0006 | wrong |
|---|--:|--:|--:|--:|
| ggml | 2.66 | 0.976 | 0.939 | 0 / 1 |
| + CoreML encoder | 3.92 | 0.939 | 0.939 | 0 / 1 |
| **+ CoreML mem-attn (both)** | **6.11** | **0.939** | **0.939** | **0 / 1** |

**~2.3× over ggml, accuracy held, wrong-person preserved.** The FP16 box-jitter
(IoU ~0.88 vs ggml) does not move `tracked_fraction` — the tracker is robust to it.

### Run it
```
cmake -S . -B build -DSAM3_TIMING=ON -DSAM3_COREML=ON
cmake --build build --target sam3_edgetam_bench -j
SAM3_COREML_ENCODER=1 SAM3_COREML_MODEL=<...>/edgetam_encoder_neck_nhwc.mlpackage \
SAM3_COREML_MEMATTN=1 SAM3_COREML_MEMATTN_MODEL=<...>/edgetam_memory_attention.mlpackage \
  ./build/examples/sam3_edgetam_bench --n-frames 40 --warmup 20
```

## What's here
- `edgetam_coreml.{h,mm}` — Objective-C++ C-ABI bridge: `edgetam_coreml_encode`
  (encoder) + `edgetam_coreml_memattn` (4-input mem-attn), load/compile a
  `.mlpackage`, predict, widen FP16→FP32, compute-unit selectable. Behind the
  `SAM3_COREML` CMake option (Apple-only, OFF by default; default builds
  unaffected) + runtime env gates `SAM3_COREML_ENCODER` / `SAM3_COREML_MEMATTN`.

## How the CoreML models are produced (external, in the eval repo)
`~/iris/audits/goldenclip-eval/`:
- encoder: export `neck(trunk(x))[0:3]` channels-last via the freeze trick
  (`torch.jit.freeze` + `run_frozen_optimizations` before `ct.convert`).
  → `edgetam_encoder_neck_nhwc.mlpackage`.
- memory attention: `convert_memattn_coreml.py` (real-valued RoPE + constant-shape
  reshapes + rank-≤5 k-rope). → `edgetam_memory_attention.mlpackage`.

## Remaining for full ~15 fps
With both legs the frame is ~164 ms (6.1 fps): encoder 11 (ANE) + mem-attn 18
(GPU) + **mask decoder ~80 (ggml)** + memory-encode ~24 (ggml) + overhead. The
**mask decoder is now the bottleneck** — it shares the propagation graph with the
mem-attn and was not separately exported. Moving the decoder (and memory-encode)
to CoreML is the remaining step toward the EdgeTAM-paper 15 fps.
