# EdgeTAM on CoreML / Apple Neural Engine (RFD 0011)

This directory holds the CoreML/ANE serving path for EdgeTAM — the route to
real-time (the EdgeTAM-paper 15 fps) on Apple silicon, which the ggml/Metal
runtime cannot reach (it is ~10× off Apple's kernels on the heavy stages).

## Measured on M4 Pro (verified, FP16, parity vs PyTorch)

| Stage | ggml/Metal | best CoreML | speedup | best device | parity |
|---|--:|--:|--:|---|--:|
| Image encoder (RepViT+FPN) | 122 ms | **12 ms** | ~10× | **ANE** | cosine 0.992 |
| Memory attention | 185 ms | **17.8 ms** | ~10.4× | **GPU** | cosine 1.000 |

**Heterogeneous placement matters:** the conv-heavy encoder is fastest on the
ANE; the attention (big matmul+softmax) is fastest on the GPU. The ANE is NOT
universally faster. Both stages converted with near-perfect parity, so a
fully-CoreML EdgeTAM frame projects to ~50–70 ms ⇒ ~15 fps.

## What's here
- `edgetam_coreml.{h,mm}` — Objective-C++ C-ABI bridge to run a CoreML encoder
  from the C++ tracker (compute-unit selectable). Built behind the `SAM3_COREML`
  CMake option (Apple-only, OFF by default) + the `SAM3_COREML_ENCODER` runtime
  env gate. **Not yet wired into `sam3_propagate_frame`** — see "Integration
  blocker" below.

## How the CoreML models are produced (external, in the eval repo)
The `.mlpackage` exports + benchmarks live with the EdgeTAM PyTorch checkout
(`~/iris/audits/goldenclip-eval/`), since they need torch + coremltools + the
EdgeTAM weights:
- encoder: `bench_encoder_ane.py`; export via the freeze trick (`torch.jit.freeze`
  + `run_frozen_optimizations` before `ct.convert` — folds the dynamic shape ops
  the stock `EdgeTAM/coreml/export_to_coreml.py` chokes on).
- memory attention: `convert_memattn_coreml.py` + `bench_memattn_coreml.py`
  (real-valued RoPE monkey-patch + constant-shape reshapes + rank-≤5 k-rope).

## Integration blocker (scoped, not yet done)
A faithful CoreML export of PyTorch EdgeTAM is NOT a drop-in for ggml's encoder:
its `backbone_fpn` high-res features are 32/64 ch, while ggml's `neck_trk` are
256 ch at every level. `vision_features` (the memory feature, 256 ch) matches and
is parity-verified; the high-res decoder features do not. Completing the swap
needs the CoreML export to emit ggml-matching 256-ch high-res features, or
sam3.cpp to consume PyTorch's 32/64-ch high-res. The memory-attention path has no
such mismatch (its I/O is well-defined and parity is 1.000).
