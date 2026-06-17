#!/usr/bin/env python3
"""RFD 0011 — measure the PURE-CoreML EdgeTAM pipeline throughput (no ggml).

Chains the 3 heavy CoreML stages with real output->input handoffs (numpy
reshapes, no cross-boundary marshalling to ggml) and times the per-frame loop.
Answers: does pure-CoreML reach ~15-20 fps vs the 6.1 fps ggml<->CoreML hybrid?

Throughput is shape-determined, so random representative inputs are valid for
the speed measurement (accuracy is separately validated at 0.939 via the hybrid,
same models). Heterogeneous placement: encoder ANE, mem-attn GPU, decoder ANE.
"""
import time
import numpy as np
import coremltools as ct

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval/coreml_models")
NE, GPU = ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU

enc = ct.models.MLModel(f"{GE}/edgetam_encoder_neck_nhwc.mlpackage", compute_units=NE)
ma  = ct.models.MLModel(f"{GE}/edgetam_memory_attention.mlpackage",  compute_units=GPU)
dec = ct.models.MLModel(f"{GE}/edgetam_mask_decoder_nhwc.mlpackage",  compute_units=NE)
print("loaded encoder(ANE) + mem-attn(GPU) + decoder(ANE)")

f32 = np.float32
img        = np.random.randn(1, 3, 1024, 1024).astype(f32)   # ImageNet-norm frame (values irrelevant for timing)
curr_pos   = np.random.randn(4096, 1, 256).astype(f32)        # fixed PE
memory     = np.random.randn(3648, 1, 64).astype(f32)         # representative 7-slot+objptr bank
memory_pos = np.random.randn(3648, 1, 64).astype(f32)
image_pe   = np.random.randn(1, 64, 64, 256).astype(f32)
sparse     = np.random.randn(1, 1, 256).astype(f32)
dense      = np.random.randn(1, 64, 64, 256).astype(f32)

def a(x): return np.ascontiguousarray(np.array(x).astype(f32))

def frame(timers=None):
    t = {}
    t0 = time.perf_counter()
    e = enc.predict({"image_norm": img})
    neck0, neck1, neck2 = a(e["neck0"]), a(e["neck1"]), a(e["neck2"])  # 256/256/256 ch, NHWC
    t["enc"] = (time.perf_counter() - t0) * 1000

    # handoff: neck2 [1,64,64,256] -> mem-attn curr [4096,1,256]
    curr = neck2.reshape(4096, 1, 256)
    t1 = time.perf_counter()
    m = ma.predict({"curr": curr, "memory": memory, "curr_pos": curr_pos, "memory_pos": memory_pos})
    t["memattn"] = (time.perf_counter() - t1) * 1000

    # handoff: var_304 [4096,1,256] -> decoder image_embeddings [1,64,64,256]
    cond = a(m["var_304"]).reshape(1, 64, 64, 256)
    t2 = time.perf_counter()
    dec.predict({"image_embeddings": cond, "image_pe": image_pe, "sparse": sparse,
                 "dense": dense, "feat_s0": neck0, "feat_s1": neck1})
    t["dec"] = (time.perf_counter() - t2) * 1000
    if timers is not None: timers.append(t)

for _ in range(10): frame()           # warmup
timers = []
T0 = time.perf_counter()
for _ in range(100): frame(timers)
wall = (time.perf_counter() - T0) * 1000

import statistics as st
def stage(k): return st.mean(x[k] for x in timers)
enc_ms, ma_ms, dec_ms = stage("enc"), stage("memattn"), stage("dec")
per = wall / 100
print(f"\n3-stage pure-CoreML chain (100 frames, after 10 warmup):")
print(f"  encoder (ANE):  {enc_ms:6.2f} ms")
print(f"  mem-attn (GPU): {ma_ms:6.2f} ms")
print(f"  decoder (ANE):  {dec_ms:6.2f} ms")
print(f"  stage sum:      {enc_ms+ma_ms+dec_ms:6.2f} ms ; glue/overhead: {per-(enc_ms+ma_ms+dec_ms):6.2f} ms")
print(f"  per-frame:      {per:6.2f} ms  =>  {1000/per:.1f} fps")
print(f"  (vs ggml<->CoreML hybrid 6.1 fps ; memory-encode not yet in this chain)")
