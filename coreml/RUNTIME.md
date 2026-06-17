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

| consumer stage | ms | unit |
|---|--:|---|
| mem-attn | 24 | GPU |
| mask decoder | 7 | ANE |
| memory-encode | **? (Egor's was 3.8)** | GPU |
| **consumer total** | **~35-43** | |

- If memory-encode exports ≈ 4 ms (Egor found it 5.75× faster on CoreML GPU than CPU —
  "the unsung hero"): consumer ≈ 35 ms → **~28 fps**. Reaches the bar.
- If memory-encode ≈ 12 ms: consumer ≈ 43 ms → **~23 fps**. Honest floor.

So **the memory-encode export is load-bearing for this runtime**, not just the Python
number — and its CoreML speed is what decides 23 vs 28 fps.

## Build pieces (in order)

1. **Memory-encode CoreML export** — `export/convert_memenc_coreml.py`. Status: the
   constant-shape perceiver patch + injected constant PEs are **done and parity-verified**
   (features cosine ≈ 1.0); `ct.convert` still hits one residual `aten::Int`
   (`only 0-dimensional arrays can be converted to Python scalars`) from a not-yet-isolated
   node. Next step: `ct.convert(..., debug=True)` (or bisect `MemEnc.forward`) to locate
   the remaining dynamic int, replace it with a constant like the others. The math is
   correct; this is convert-plumbing, not algorithm.
2. **Host-side memory bank + glue** — port off ggml: the `num_maskmem` ring, obj-pointer
   management, and the U3 SAMURAI motion-aware selection (currently in `sam3.cpp`) onto
   the CoreML/host side, feeding the mem-attn model's fixed-capacity inputs.
3. **C++ pipeline harness** — two `std::thread`s + a bounded SPSC queue (depth 2);
   resident `MLMultiArray`s; encoder on ANE ahead of the GPU/ANE consumer. Reuse
   `edgetam_coreml_{encode,memattn,decode}` from `edgetam_coreml.mm`; add
   `edgetam_coreml_memencode`.
4. **cgo into platform (U8)** — C-ABI surface, proto/State-Sync/NATS byte-identical gate.

## What's already done (this PR is the foundation)
- 3 of 4 stage models exported + parity-verified (encoder 0.992, mem-attn 1.000,
  decoder 0.999); the 4th is plumbing away (above).
- The Obj-C++ bridge with resident-model load + FP16 widening + compute-unit selection.
- The hybrid proving the stages are correct end-to-end (goldeneval 0.939, wrong-person 0).
- The measured ceilings (sequential 19, threaded 23.7) that set the runtime's target.
