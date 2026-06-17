# Pure-CoreML C++ runtime — design for the real-time path (RFD 0011, next PR)

This is the remaining build that reaches Egor Dmitriev's ~28-30 fps. It is **scoped
here, not implemented in this PR** — it is a new subsystem (its own PR), and this PR
is already large. Everything below is turnkey: the parity-verified stage models, the
measured bottleneck, and the one blocked export are all specified.

## Why a new runtime (not more of the hybrid)

The shipping config is the **hybrid**: ggml tracker with two stages on CoreML (encoder
ANE, mem-attn GPU) = 6.1 fps, accuracy held. Its ceiling is the ggml↔CoreML marshalling
(84 MB/frame decoder inputs copied each `predict()`), measured at 0.17 ms glue in the
pure-CoreML chain — i.e. the loss is the boundary, not compute.

The Python pure-CoreML pipeline removes the ggml boundary and hits 19 fps sequential.
Threading it (`bench/pipeline_coreml_threaded.py`) reaches only **23.7 fps (1.24×)**
because `coremltools.predict()` releases the GIL only during kernel execution and holds
it through CPU-side marshalling — two predicts mostly serialize. **A C++ runtime has no
GIL**, can keep `MLMultiArray`s resident across calls, and can dispatch stages on GCD /
`std::thread` for genuine cross-unit concurrency. The Obj-C++ bridge
(`edgetam_coreml.mm`) is already the foundation.

## Architecture — encoder-ahead pipeline (Egor's shape)

The memory bank creates a hard serial dependency: frame N's mem-attn reads memory written
by frame N-1's memory-encode. So the **consumer is serial**; only the encoder (which
depends on nothing but the raw frame) can run ahead.

```
Thread A (producer):  encoder[N+1]   on ANE              ─┐ Queue(2)
Thread B (consumer):  mem-attn[N] → decoder[N] →          │
                      memory-encode[N] → SAMURAI/Kalman   ─┘  on GPU/ANE
```

Throughput is **consumer-bound** (encoder hides behind it). Measured/expected per-stage
on M4 Pro:

| consumer stage | ms (measured) | unit |
|---|--:|---|
| mem-attn | 18-24 | GPU |
| mask decoder | 6-7 | ANE |
| memory-encode | **5.0** | ANE/GPU |
| **consumer total** | **~29-36** | |

Memory-encode is now exported and **measured at ~5 ms** (close to Egor's 3.8 ms "unsung
hero") — not the 10-15 ms feared. So the consumer is ~29-36 ms and, with the encoder hidden
one frame ahead, the runtime is **consumer-bound at ~28-34 fps**. The full 4-stage
*sequential* chain already measures **20.3 fps** (`bench/pipeline_coreml_4stage.py`);
pipelining the encoder ahead of the consumer is the lever from 20 → ~28-34.

## Build pieces (in order)

1. **Memory-encode CoreML export** — `export/convert_memenc_coreml.py`. **DONE.**
   Constant-shape perceiver windowing + head-split + injected constant PEs; exported with
   parity cosine **1.000000**, ~5 ms on ANE/GPU. → `edgetam_memory_encode.mlpackage`. All
   four stage models now exist; the runtime is glue + threading, no more export blockers.
2. **Host-side memory bank + glue** — port off ggml: the `num_maskmem` ring, obj-pointer
   management, and the U3 SAMURAI motion-aware selection (currently in `sam3.cpp`) onto
   the CoreML/host side, feeding the mem-attn model's fixed-capacity inputs.
3. **C++ pipeline harness** — two `std::thread`s + a bounded SPSC queue (depth 2);
   resident `MLMultiArray`s; encoder on ANE ahead of the GPU/ANE consumer. Reuse
   `edgetam_coreml_{encode,memattn,decode}` from `edgetam_coreml.mm`; add
   `edgetam_coreml_memencode`.
4. **cgo into platform (U8)** — C-ABI surface, proto/State-Sync/NATS byte-identical gate.

## What's already done (this PR is the foundation)
- All 4 stage models exported + parity-verified (encoder 0.992, mem-attn 1.000,
  decoder 0.999, memory-encode 1.000) and benched.
- The Obj-C++ bridge with resident-model load + FP16 widening + compute-unit selection.
- The hybrid proving the stages are correct end-to-end (goldeneval 0.939, wrong-person 0).
- The measured ceilings (sequential 19, threaded 23.7) that set the runtime's target.
