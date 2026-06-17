# EdgeTAM on CoreML / Apple Neural Engine (RFD 0011)

The CoreML/ANE serving path for EdgeTAM — the route to real-time on Apple
silicon, which the ggml/Metal runtime cannot reach (ggml is ~10× off Apple's
kernels on the heavy stages). **The CoreML/ANE image encoder is wired into the
tracker and working** (`SAM3_COREML` build + `SAM3_COREML_ENCODER` runtime gate).

## Measured on M4 Pro (verified, FP16, parity vs PyTorch)

| Stage | ggml/Metal | best CoreML | speedup | best device | parity |
|---|--:|--:|--:|---|--:|
| Image encoder (RepViT+FPN) | 122 ms | **11 ms** | ~11× | **ANE** | cosine 0.992 |
| Memory attention | 185 ms | **17.8 ms** | ~10.4× | **GPU** | cosine 1.000 |

**Heterogeneous placement matters:** the conv-heavy encoder is fastest on the
ANE; the attention (big matmul+softmax) is fastest on the GPU. The ANE is NOT
universally faster.

## Live hybrid encoder (DONE)
The CoreML/ANE encoder replaces the ggml RepViT+FPN inside `edgetam_encode_image`:
- **End-to-end: ~2.6 → 3.92 fps (~1.5×)** on the hold loop; `image_encoder_compute`
  122 → 11 ms (ANE); `state_update` 1.6 ms (the CoreML neck output is exported
  **channels-last `[1,H,W,D]`**, byte-identical to ggml `neck_trk [D,W,H]`, so the
  load is a `memcpy` — no per-frame transpose).
- **Accuracy holds** (goldeneval, seed-once, tf@0.5): dancetrack0004 0.939,
  dancetrack0006 0.939 (vs the ggml encoder's 0.976 / 0.939; wrong-person frames
  0 / 1 — preserved). The small dt0004 dip is FP16 feature jitter; an FP32 export
  closes it at some ANE cost.

### Run it
```
cmake -S . -B build -DSAM3_TIMING=ON -DSAM3_COREML=ON
cmake --build build --target sam3_edgetam_bench -j
SAM3_COREML_ENCODER=1 \
SAM3_COREML_MODEL=<...>/edgetam_encoder_neck_nhwc.mlpackage \
  ./build/examples/sam3_edgetam_bench --n-frames 30 --warmup 5
```

## What's here
- `edgetam_coreml.{h,mm}` — Objective-C++ C-ABI bridge (load/compile a
  `.mlpackage`, predict, widen FP16→FP32, compute-unit selectable). Built behind
  the `SAM3_COREML` CMake option (Apple-only, OFF by default); default builds
  unaffected. The encoder path in `sam3.cpp` is additionally gated at runtime by
  `SAM3_COREML_ENCODER` + `SAM3_COREML_MODEL`.

## How the CoreML models are produced (external, in the eval repo)
The exports + benchmarks live with the EdgeTAM PyTorch checkout
(`~/iris/audits/goldenclip-eval/`):
- encoder: export `neck(trunk(x))[0:3]` (the 256-ch fused FPN levels that match
  ggml `neck_trk`), channels-last, via the freeze trick (`torch.jit.freeze` +
  `run_frozen_optimizations` before `ct.convert` — folds the dynamic shape ops the
  stock `EdgeTAM/coreml/export_to_coreml.py` chokes on).
  → `edgetam_encoder_neck_nhwc.mlpackage`.
- memory attention: `convert_memattn_coreml.py` + `bench_memattn_coreml.py`
  (real-valued RoPE monkey-patch + constant-shape reshapes + rank-≤5 k-rope).

## Remaining for full ~15 fps
The encoder is wired. The memory attention is **converted + benchmarked**
(185 → 17.8 ms GPU, cosine 1.000) but **not yet wired** — its hot inputs (the
memory bank + RoPE tensors) must be marshalled to CoreML each frame. Wiring it
(encoder ANE ‖ mem-attn GPU) is the remaining step to the EdgeTAM-paper 15 fps;
the hard part (conversion) is done.
