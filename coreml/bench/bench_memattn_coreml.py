# Benchmark EdgeTAM memory_attention CoreML model per compute unit + parity vs PyTorch.
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
import coremltools as ct

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval")
PKG = f"{GE}/coreml_models/edgetam_memory_attention.mlpackage"

d = np.load("/tmp/memattn_inputs.npz")
feeds = {k: d[k].astype(np.float32) for k in ["curr", "memory", "curr_pos", "memory_pos"]}
ref = np.load("/tmp/memattn_ref_out.npy")  # PyTorch complex-RoPE reference output [4096,1,256]

def cosine(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

units = {
    "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE":  ct.ComputeUnit.CPU_AND_NE,
    "ALL":         ct.ComputeUnit.ALL,
}

print(f"Benchmarking {PKG}")
print(f"Inputs: " + ", ".join(f"{k}{v.shape}" for k, v in feeds.items()))
print(f"{'compute_unit':<14} {'mean_ms':>9} {'std_ms':>8} {'min_ms':>8}   {'cosine_vs_torch':>16}")
print("-" * 70)

out_name = None
for uname, u in units.items():
    try:
        m = ct.models.MLModel(PKG, compute_units=u)
        if out_name is None:
            out_name = list(m.output_description)[0]
        # warmup
        for _ in range(5):
            r = m.predict(feeds)
        out = r[out_name]
        cos = cosine(out, ref)
        # timed
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            m.predict(feeds)
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        print(f"{uname:<14} {ts.mean():>9.2f} {ts.std():>8.2f} {ts.min():>8.2f}   {cos:>16.6f}")
    except Exception as e:
        print(f"{uname:<14} FAILED: {type(e).__name__}: {str(e)[:80]}")

print("-" * 70)
print("Reference: ggml/Metal mem_attn = 185 ms/frame (the bottleneck being targeted).")
