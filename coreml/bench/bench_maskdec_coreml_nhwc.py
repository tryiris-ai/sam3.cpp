# Benchmark the CHANNELS-LAST (NHWC) EdgeTAM mask decoder CoreML model.
# Same as bench_maskdec_coreml.py but: (1) points at the _nhwc.mlpackage, (2) reads
# the NHWC /tmp inputs+refs the NHWC converter saved, (3) benches the two compute
# units the task targets -- CPU_AND_NE (ANE) and CPU_AND_GPU. 5 warmup + 20 timed.
# Parity is cosine vs the UNPATCHED PyTorch wrapper on the identical NHWC tensors.
# Reference: ggml/Metal mask decoder ~80 ms/frame (the last bottleneck stage).
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
import coremltools as ct

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval")
PKG = f"{GE}/coreml_models/edgetam_mask_decoder_nhwc.mlpackage"

# Same fixed NHWC inputs + PyTorch reference the NHWC converter saved.
din = np.load("/tmp/maskdec_nhwc_inputs.npz")
order = ["image_embeddings", "image_pe", "sparse", "dense", "feat_s0", "feat_s1"]
feeds = {k: din[k].astype(np.float32) for k in order}
dref = np.load("/tmp/maskdec_nhwc_ref.npz")
ref = {n: dref[n] for n in ["masks", "iou_pred", "obj_score", "mask_tokens"]}


def cosine(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Task targets ANE + GPU; include ALL/CPU_ONLY for context but report focuses on the two.
units = {
    "CPU_AND_NE":  ct.ComputeUnit.CPU_AND_NE,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "ALL":         ct.ComputeUnit.ALL,
    "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
}

print(f"Benchmarking {PKG}")
print(f"Inputs (NHWC): " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in feeds.items()))
print(f"{'compute_unit':<14} {'mean_ms':>9} {'std_ms':>8} {'min_ms':>8}   "
      f"{'cos_masks':>10} {'cos_iou':>9} {'cos_obj':>9} {'cos_tok':>9}")
print("-" * 92)

for uname, u in units.items():
    try:
        m = ct.models.MLModel(PKG, compute_units=u)
        # warmup (5)
        for _ in range(5):
            r = m.predict(feeds)
        cm = cosine(r["masks"], ref["masks"])
        ci = cosine(r["iou_pred"], ref["iou_pred"])
        co = cosine(r["obj_score"], ref["obj_score"])
        ct_ = cosine(r["mask_tokens"], ref["mask_tokens"])
        # timed (20)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            m.predict(feeds)
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        print(f"{uname:<14} {ts.mean():>9.2f} {ts.std():>8.2f} {ts.min():>8.2f}   "
              f"{cm:>10.6f} {ci:>9.6f} {co:>9.6f} {ct_:>9.6f}")
    except Exception as e:
        print(f"{uname:<14} FAILED: {type(e).__name__}: {str(e)[:80]}")

print("-" * 92)
print("Reference: ggml/Metal mask decoder ~80 ms/frame (the last bottleneck stage).")
