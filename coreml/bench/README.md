# CoreML serving-path tooling (RFD 0011)

The exact scripts that **produce** and **measure** the EdgeTAM CoreML models in this
PR. Vendored from the goldenclip-eval harness for reproducibility. They run against
the external EdgeTAM checkpoint + the exported `.mlpackage`s, not the ggml runtime —
point them at your copy with `SAM3_GE_DIR` (defaults to the author's eval dir).

```
export SAM3_GE_DIR=/path/to/goldenclip-eval     # holds coreml_models/, EdgeTAM/, edgetam_ckpt/
python -m venv .venv && .venv/bin/pip install coremltools torch timm hydra-core
```

## export/ — produce the `.mlpackage` models (need torch + EdgeTAM ckpt)
| script | stage | technique |
|---|---|---|
| `convert_memattn_coreml.py` | memory attention | real-valued RoPE + constant-shape `_separate_heads` + rank-≤5 k-rope |
| `convert_maskdec_coreml_nhwc.py` | mask decoder | channels-last NHWC outputs (byte-identical to ggml) |
| `convert_memenc_coreml.py` | memory-encode (stage 4) | constant-shape perceiver windowing + injected constant PEs — **parity OK, convert blocked, see ../RUNTIME.md** |

The image-encoder export (`neck(trunk(x))[0:3]` channels-last via the `jit.freeze` +
`run_frozen_optimizations` trick) lives in the eval harness; the produced
`edgetam_encoder_neck_nhwc.mlpackage` is the input to the bench scripts.

## bench/ — measure (need only coremltools + the exported models)
| script | what it measures | headline |
|---|---|---|
| `encoder_compute_sweep.py` | encoder p50 across CPUOnly/ANE/GPU/ALL | ANE 16.7 ms wins on M4 Pro (2.7× over CPUOnly) |
| `pipeline_coreml.py` | 3-stage **sequential** pure-CoreML chain | 19.1 fps, 0.17 ms glue |
| `pipeline_coreml_threaded.py` | 3-stage **threaded** pipeline + GIL/overlap probe (`SAM3_DEC_UNIT=ane\|gpu\|cpu`) | 23.7 fps (dec→GPU), 1.24× — GIL-bound |
| `bench_encoder_ane.py` / `bench_memattn_coreml.py` / `bench_maskdec_coreml_nhwc.py` | per-stage isolation timings | see `coreml/README.md` table |

```
SAM3_GE_DIR=... python bench/encoder_compute_sweep.py
SAM3_GE_DIR=... python bench/pipeline_coreml.py
SAM3_GE_DIR=... SAM3_DEC_UNIT=gpu python bench/pipeline_coreml_threaded.py
```

`.mlpackage` binaries are NOT committed (tens of MB each); regenerate via `export/` or
copy from the eval `coreml_models/` dir.
