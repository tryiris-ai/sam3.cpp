#!/usr/bin/env python3
"""RFD 0011 — benchmark the EdgeTAM CoreML image encoder per compute unit.

Answers the concrete question from PABannier's tweet ("compile the graph and
serve it via CoreML to run it on the ANE"): how fast is the EdgeTAM RepViT+FPN
image encoder on the Apple Neural Engine vs the GPU vs CPU — and how does the
ANE number compare to our measured ggml/Metal encode (~122 ms)?

Loads the exported .mlpackage under each ct.ComputeUnit and times predict().
CPU_AND_NE is the ANE-eligible path. Also prints the CoreML compute plan
(which device each op was actually dispatched to) when available, so we can see
whether the encoder genuinely lands on the ANE or falls back.
"""
import sys, time
import numpy as np
import coremltools as ct
from PIL import Image

mlpkg = sys.argv[1] if len(sys.argv) > 1 else "coreml_models/edgetam_image_encoder.mlpackage"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# Encoder input is an ImageType (1,3,1024,1024), RGB, scale 1/255 baked in.
img = Image.fromarray((np.random.rand(1024, 1024, 3) * 255).astype("uint8"))

UNITS = [
    ("CPU_ONLY",    ct.ComputeUnit.CPU_ONLY),
    ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
    ("CPU_AND_NE",  ct.ComputeUnit.CPU_AND_NE),   # ANE-eligible
    ("ALL",         ct.ComputeUnit.ALL),
]

print(f"model: {mlpkg}")
print(f"runs:  {N} (after 5 warmup)\n")
print(f"  {'compute_unit':<14}{'mean ms':>10}{'p50 ms':>9}  vs ggml/Metal 122ms")
print("  " + "-" * 56)
results = {}
for name, unit in UNITS:
    try:
        m = ct.models.MLModel(mlpkg, compute_units=unit)
        for _ in range(5):
            m.predict({"image": img})
        ts = []
        for _ in range(N):
            t0 = time.perf_counter()
            m.predict({"image": img})
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        mean = sum(ts) / len(ts)
        p50 = ts[len(ts) // 2]
        results[name] = mean
        speedup = 122.0 / mean
        print(f"  {name:<14}{mean:>10.1f}{p50:>9.1f}  {speedup:>5.2f}x")
    except Exception as e:
        print(f"  {name:<14}  FAILED: {str(e)[:60]}")

# Compute-plan: which backend each op actually ran on (coremltools >= 7.2).
print()
try:
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        mlpkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    from collections import Counter
    dev = Counter()
    prog = plan.model_structure.program
    for fname, func in prog.functions.items():
        for op in func.block.operations:
            di = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if di is not None:
                dev[type(di.preferred_compute_device).__name__] += 1
    print("  op dispatch (CPU_AND_NE):", dict(dev))
except Exception as e:
    print(f"  (compute plan unavailable: {str(e)[:80]})")
