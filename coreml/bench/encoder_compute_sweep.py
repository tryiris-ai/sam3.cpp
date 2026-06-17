#!/usr/bin/env python3
"""RFD 0011 / U5 prep — encoder compute-unit sweep on THIS M4 Pro.

Egor's part-3 most surprising finding: the ViT image_encoder was fastest on
CPUOnly (BNNS/AMX), beating BOTH GPU and ANE (his M1 Pro: 23.7 CPUOnly vs 29.6
ANE vs 44 GPU). We placed the encoder on ANE and never benchmarked CPUOnly.

This settles it on our hardware AND decides the 3-stage pipeline topology: if the
encoder leaves the ANE (onto CPU/BNNS), encoder/mem-attn/decoder land on three
distinct compute units (CPU, GPU, ANE) = clean overlap, exactly Egor's shape.
If ANE wins, encoder+decoder share the ANE and contend.

numpy + coremltools only (no torch/cv2 import → avoids the known SIGSEGV).
Throughput is shape-determined, so random representative input is valid for timing.
"""
import time, statistics as st
import numpy as np
import coremltools as ct

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval/coreml_models")
ENC = f"{GE}/edgetam_encoder_neck_nhwc.mlpackage"

UNITS = [
    ("CPUOnly",          ct.ComputeUnit.CPU_ONLY),
    ("CPUAndNeuralEngine", ct.ComputeUnit.CPU_AND_NE),
    ("CPUAndGPU",        ct.ComputeUnit.CPU_AND_GPU),
    ("ALL",              ct.ComputeUnit.ALL),
]
f32 = np.float32
img = np.random.randn(1, 3, 1024, 1024).astype(f32)   # ImageNet-norm frame (values irrelevant for timing)
WARMUP, N = 12, 50

print(f"encoder = {ENC.split('/')[-1]}   warmup={WARMUP} timed={N}\n")
print(f"{'compute unit':22} {'p50 ms':>8} {'p95 ms':>8} {'min ms':>8}")
print("-" * 50)
results = {}
for name, unit in UNITS:
    m = ct.models.MLModel(ENC, compute_units=unit)
    for _ in range(WARMUP):
        m.predict({"image_norm": img})          # first call compiles for this unit
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        m.predict({"image_norm": img})
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    p50, p95, mn = st.median(ts), ts[int(0.95 * len(ts))], ts[0]
    results[name] = p50
    print(f"{name:22} {p50:8.2f} {p95:8.2f} {mn:8.2f}")
    del m

best = min(results, key=results.get)
print(f"\nfastest encoder placement on this box: {best} ({results[best]:.2f} ms p50)")
print(f"(ANE = CPUAndNeuralEngine = {results['CPUAndNeuralEngine']:.2f} ms ; "
      f"current pipeline uses ANE)")
