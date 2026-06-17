# Pure-CoreML C++ runtime — core BUILT + MEASURED (RFD 0011)

The real-time path. The **runtime core is now built and measured** in this PR
(`examples/sam3_coreml_pipeline.cpp` + the `edgetam_coreml_memencode` bridge); the
remaining pieces (real host-side memory bank, cgo) are scoped below. The headline is a
correction: **the GIL was not the dominant bottleneck**, and a no-GIL C++ pipeline does
*not* deliver a clean ~2× or a stable 28-30 fps — placement and thermals dominate.

## Why a C++ runtime (the original thesis)

The Python pure-CoreML pipeline hits 19 fps sequential / 23.7 fps threaded;
`coremltools.predict()` holds the GIL through its CPU-side marshalling, so threads mostly
serialize (~1.15× overlap). The thesis was that a no-GIL C++ pipeline (encoder one frame
ahead of the consumer, on separate silicon) would reach the consumer-bound ~28-34 fps.

## What got built (this PR)

- `edgetam_coreml_memencode` — the 4th-stage bridge call (memory_encoder + perceiver),
  parity-verified, ~5 ms.
- `examples/sam3_coreml_pipeline.cpp` — a 4-stage **encoder-ahead** pipeline: a producer
  thread runs the encoder; a consumer thread runs mem-attn → decoder → memory-encode;
  connected by a bounded 3-slot pool. Each `MLModel` is owned by exactly one thread (no
  concurrent access). Fixed/representative memory bank (throughput is shape-determined).
- CMake: builds under `-DSAM3_COREML=ON`. (Bridge `.mm` compiled as CXX with
  `-x objective-c++` — `enable_language(OBJCXX)` fails to emit a compile rule under CMake
  4.x in that sub-scope.)

## What it measured (M4 Pro) — and the two findings

**Finding 1 — placement is decisive; threading HELPS only with silicon separation.**

| config | threaded vs its sequential |
|---|--:|
| encoder ANE, consumer dec+memenc also on **ANE** (3 stages on ANE) | **0.62× — slower!** |
| encoder ANE ∥ consumer **all-GPU** (dec+memenc→GPU) | **1.1-1.35× faster** |

With the encoder and the consumer both leaning on the ANE, threading just thrashes the one
unit and *regresses*. Only when the encoder (ANE) and the whole consumer (GPU) sit on
different silicon does overlap appear — and even then it is a modest 1.1-1.35×, not 2×
(the two threads still contend on CPU-side input marshalling + unified-memory bandwidth).

**Finding 2 — absolute throughput is thermal- and mode-switch-bound, not stable.**
Best silicon-separated config across runs: **~27 fps cold (first run), ~14-18 fps sustained**
(back-to-back runs as the Mac heats). Same per-stage times throughout, ~2× swing in chain
throughput — classic thermal throttling (Egor hit this too) plus the **per-frame ANE↔GPU
mode switch**, which makes the real chain run ~2× the sum of warm isolated per-stage times.

**Corrected conclusion:** removing the GIL did not unlock a stable 28-30 fps. The real caps
are (a) per-frame ANE↔GPU transitions, (b) CPU-side marshalling contention between threads
(the bridge alloc+memcpys every input each call), and (c) thermal throttling. C++ threading
is a ~1.1-1.35× lever *with correct placement* and a regression without — not the projected
~2×. The honest sustained number for this build is **~15-20 fps**, peaking ~27 cold.

## Remaining levers (to actually stabilize real-time)
1. **Resident `MLMultiArray`s** — the bridge currently allocates + memcpys every input on
   each call; keeping inputs resident and writing in place would cut the CPU-side contention
   that caps the overlap at 1.1×. Likely the single biggest lever left.
2. **Fewer unit transitions** — the per-frame ANE↔GPU switch is expensive; an all-GPU chain
   (encoder GPU too) trades the ANE's raw speed for no switch tax and may be *more stable*
   (worth measuring) even if lower-peak.
3. **Thermal headroom** — sustained clocks matter; report steady-state, not cold peaks.
4. **Host-side memory bank** — port the `num_maskmem` ring + obj-pointers + U3 SAMURAI
   selection off ggml so the pipeline drives a real tracker (this build used a fixed bank).
5. **cgo into platform (U8)** — C-ABI surface, proto/State-Sync/NATS byte-identical gate.

## What's already solid (the foundation)
- All 4 stage models exported + parity-verified (encoder 0.992, mem-attn 1.000,
  decoder 0.999, memory-encode 1.000) and benched.
- The Obj-C++ bridge (4 stage calls) + the threaded pipeline harness, building + running.
- The hybrid proving the stages are correct end-to-end (goldeneval 0.939, wrong-person 0).
- The measured truth that sets honest expectations: ~15-20 fps sustained, placement-critical.
