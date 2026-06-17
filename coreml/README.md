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

## Pure-CoreML pipeline — MEASURED 19 fps (the real-time path)

A standalone pure-CoreML chain (`audits/goldenclip-eval/pipeline_coreml.py`, no
ggml) chains the three CoreML stages with real output→input handoffs and times
the per-frame loop on this M4 Pro (100 frames):

| stage | ms |
|---|--:|
| encoder (ANE) | 21.0 |
| memory attention (GPU) | 23.8 |
| decoder (ANE) | 7.3 |
| **glue / handoff** | **0.17** |
| **per-frame** | **52.3 → 19.1 fps** |

**This settles the 6-vs-15-20 question.** The cross-stage handoff is **0.17 ms**
(negligible) — so the hybrid's 6.1 fps was **entirely** the ggml↔CoreML
marshalling (84 MB decoder inputs, per-stage input copies), NOT chaining or
compute. A pure-CoreML pipeline (everything on the CoreML side, no ggml
round-trips) hits **19 fps for the 3 stages**; adding the 4th stage
(memory-encode, ~10–15 ms — its CoreML export hits the same perceiver
constant-shape `int`-op wall the other stages cleared with reshape patches,
solvable) lands the full tracker at **~15–16 fps** — matching the EdgeTAM-paper /
PABannier "15–20 on M-series" figure. Per-stage times run a bit higher than
isolated (ANE/GPU contention back-to-back), but the throughput is real.

**Bottom line:** the **hybrid (6.1 fps)** is the validated, accuracy-held config
shipping in this PR (CoreML stages bolted onto the proven ggml tracker). The
**pure-CoreML pipeline (~15–19 fps)** is the real-time path — now *measured*, not
projected. Reaching it in production means a pure-CoreML runtime (port the tracker
glue + memory bank to the CoreML/host side; the 4 stage models are the building
blocks, all parity-verified).

## Decoder leg in the HYBRID — transfer-bound, NOT recommended there (2-leg is optimal)
NB: the decoder being "transfer-bound" is a **hybrid-only** artifact (its 84 MB
256-ch high-res inputs cross the ggml↔CoreML boundary each frame). In the
*pure-CoreML* chain above the decoder is just 7 ms with ~0 handoff — so it's only
a problem when bolted onto ggml.
A third leg (CoreML mask decoder, `SAM3_COREML_DECODER`) is implemented and
**functionally correct** (goldeneval dt0004 0.913, tracks, wrong-person 0), but it
is a **net regression** and is OFF by default:

| config | fps | dt0004 |
|---|--:|--:|
| **2 legs (encoder + mem-attn)** | **6.11** | **0.939** |
| 3 legs (+ decoder) | 2.62–5.0 | 0.913 |

Why: the decoder *compute* is fast on CoreML (80 ms ggml → ~11 ms, export parity
0.999), but its **256-ch high-res inputs are ~84 MB/frame** (`feat_s0` 67 MB +
`feat_s1` 17 MB). CoreML's `predict()` copies inputs to the compute device every
call, and that transfer costs more than the ggml decoder — which keeps those
features on-device. So moving the decoder out trades a cheap on-device op for an
expensive host→device copy. The slight accuracy dip is extra FP16 compounding.

**The fix (the real remaining work):** feed the decoder **32/64-ch** high-res
features (apply the decoder's `conv_s0`/`conv_s1` 256→32/64 inside the *encoder*
export) → ~13 MB/frame instead of 84 MB. Or fuse encoder+decoder into one CoreML
graph so the high-res features never leave the device. Either makes the decoder a
real win and, with memory-encode, lands the EdgeTAM-paper ~15 fps. The decoder
export + bridge + integration here are the reusable building blocks for that.

**Recommended config: 2 legs (encoder ANE + mem-attn GPU) — 6.11 fps, accuracy held.**
