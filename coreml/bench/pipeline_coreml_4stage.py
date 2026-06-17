#!/usr/bin/env python3
"""RFD 0011 — FULL 4-stage pure-CoreML pipeline now that memory-encode exports.

Stage 4 (memory_encoder + spatial_perceiver) is exported (parity 1.000000), so the
"~15-16 fps estimate" for the full tracker can be replaced with a measurement. Sweeps
memory-encode placement (Egor found it fastest on GPU — "the unsung hero", 3.8 ms) and
times the 4-stage sequential chain. numpy + coremltools only.
"""
import time, statistics as st
import numpy as np
import coremltools as ct

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval/coreml_models")
NE, GPU, CPU = ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_ONLY
f32 = np.float32
def a(x): return np.ascontiguousarray(np.array(x).astype(f32))

img        = np.random.randn(1, 3, 1024, 1024).astype(f32)
curr_pos   = np.random.randn(4096, 1, 256).astype(f32)
memory     = np.random.randn(3648, 1, 64).astype(f32)
memory_pos = np.random.randn(3648, 1, 64).astype(f32)
image_pe   = np.random.randn(1, 64, 64, 256).astype(f32)
sparse     = np.random.randn(1, 1, 256).astype(f32)
dense      = np.random.randn(1, 64, 64, 256).astype(f32)
mask_full  = np.random.randn(1, 1, 1024, 1024).astype(f32)

enc = ct.models.MLModel(f"{GE}/edgetam_encoder_neck_nhwc.mlpackage", compute_units=NE)
ma  = ct.models.MLModel(f"{GE}/edgetam_memory_attention.mlpackage",  compute_units=GPU)
dec = ct.models.MLModel(f"{GE}/edgetam_mask_decoder_nhwc.mlpackage",  compute_units=NE)
print("loaded enc(ANE) + memattn(GPU) + dec(ANE) + memory-encode")

# ---- memory-encode compute-unit sweep (which silicon, like Egor's "unsung hero") ----
print("\n[memory-encode compute-unit sweep]")
me_best, me_best_ms = None, 1e9
for name, unit in [("CPUOnly", CPU), ("ANE", NE), ("GPU", GPU)]:
    m = ct.models.MLModel(f"{GE}/edgetam_memory_encode.mlpackage", compute_units=unit)
    for _ in range(8): m.predict({"pix_feat": np.random.randn(1,256,64,64).astype(f32), "mask_logits": mask_full})
    ts = []
    for _ in range(30):
        pf = np.random.randn(1,256,64,64).astype(f32)
        t0 = time.perf_counter(); m.predict({"pix_feat": pf, "mask_logits": mask_full}); ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); p50 = st.median(ts)
    print(f"  {name:8} {p50:6.2f} ms")
    if p50 < me_best_ms: me_best_ms, me_best, me_unit = p50, name, unit

print(f"  -> memory-encode best: {me_best} ({me_best_ms:.2f} ms)")
me = ct.models.MLModel(f"{GE}/edgetam_memory_encode.mlpackage", compute_units=me_unit)

# ---- 4-stage sequential chain (real handoffs where chainable; representative for memenc inputs) ----
def frame(timers=None):
    t = {}
    t0 = time.perf_counter(); e = enc.predict({"image_norm": img})
    n0, n1, n2 = a(e["neck0"]), a(e["neck1"]), a(e["neck2"]); t["enc"] = (time.perf_counter()-t0)*1000
    curr = n2.reshape(4096, 1, 256)
    t1 = time.perf_counter(); m = ma.predict({"curr": curr, "memory": memory, "curr_pos": curr_pos, "memory_pos": memory_pos}); t["ma"] = (time.perf_counter()-t1)*1000
    cond = a(m["var_304"]).reshape(1, 64, 64, 256)
    t2 = time.perf_counter(); dec.predict({"image_embeddings": cond, "image_pe": image_pe, "sparse": sparse, "dense": dense, "feat_s0": n0, "feat_s1": n1}); t["dec"] = (time.perf_counter()-t2)*1000
    # stage 4: pix_feat = encoder 64x64 feature (neck2 -> BCHW); mask = decoder full-res (representative)
    pix_feat = np.ascontiguousarray(n2.reshape(64, 64, 256).transpose(2, 0, 1)[None]).astype(f32)
    t3 = time.perf_counter(); me.predict({"pix_feat": pix_feat, "mask_logits": mask_full}); t["me"] = (time.perf_counter()-t3)*1000
    if timers is not None: timers.append(t)

for _ in range(10): frame()
timers = []; T0 = time.perf_counter()
for _ in range(100): frame(timers)
wall = (time.perf_counter()-T0)*1000
def stage(k): return st.mean(x[k] for x in timers)
per = wall/100
print(f"\n4-stage pure-CoreML SEQUENTIAL chain (100 frames):")
print(f"  encoder (ANE):       {stage('enc'):6.2f} ms")
print(f"  mem-attn (GPU):      {stage('ma'):6.2f} ms")
print(f"  decoder (ANE):       {stage('dec'):6.2f} ms")
print(f"  memory-encode ({me_best}): {stage('me'):6.2f} ms")
print(f"  per-frame:           {per:6.2f} ms  =>  {1000/per:.1f} fps")
print(f"  (was estimated ~15-16 fps with memory-encode ~10-15ms; now MEASURED)")
