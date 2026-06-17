#!/usr/bin/env python3
"""RFD 0011 / U5 — 3-stage THREADED pure-CoreML pipeline (Egor part-3 topology).

Converts the verified-correct 19 fps *sequential* chain (pipeline_coreml.py) into
the real-time pipelined number by overlapping stages across frames on distinct
compute units, exactly as Egor does (preprocess ∥ encoder ∥ consumer via bounded
queues). Here the three CoreML stages are the pipeline:

    [encoder ANE] --Queue(2)--> [mem-attn GPU] --Queue(2)--> [decoder ANE]

The whole premise is that the stages run on physically separate silicon and
overlap. Egor's threads overlap because ORT releases the GIL during run(); for
`coremltools.predict()` that is NOT guaranteed, so PROBE it first — if predict
holds the GIL, Python threading serializes and the pipelined number is a lie.
(Production runs this in the C++/Obj-C++ bridge where there is no GIL at all;
the Python harness only validates the throughput model.)

numpy + coremltools + threading + queue only (no torch/cv2 → avoids the SIGSEGV).
Throughput is shape-determined, so random representative inputs are valid for
timing; accuracy is separately validated at 0.939 (hybrid, same models).
"""
import os, time, threading, queue, statistics as st
import numpy as np
import coremltools as ct

import os
GE = os.environ.get("SAM3_GE_DIR", "/Users/noah.johnson/iris/audits/goldenclip-eval/coreml_models")
NE  = ct.ComputeUnit.CPU_AND_NE
GPU = ct.ComputeUnit.CPU_AND_GPU
DEC_UNIT = {"ane": NE, "gpu": GPU, "cpu": ct.ComputeUnit.CPU_ONLY}[os.environ.get("SAM3_DEC_UNIT", "ane")]

f32 = np.float32
def a(x): return np.ascontiguousarray(np.array(x).astype(f32))   # widen F16 outputs

# Fixed/representative inputs (shapes from pipeline_coreml.py + edgetam_coreml.mm)
img        = np.random.randn(1, 3, 1024, 1024).astype(f32)
curr_pos   = np.random.randn(4096, 1, 256).astype(f32)
memory     = np.random.randn(3648, 1, 64).astype(f32)
memory_pos = np.random.randn(3648, 1, 64).astype(f32)
image_pe   = np.random.randn(1, 64, 64, 256).astype(f32)
sparse     = np.random.randn(1, 1, 256).astype(f32)
dense      = np.random.randn(1, 64, 64, 256).astype(f32)

print("loading encoder(ANE) + mem-attn(GPU) + decoder(%s)" % os.environ.get("SAM3_DEC_UNIT", "ane"))
enc = ct.models.MLModel(f"{GE}/edgetam_encoder_neck_nhwc.mlpackage", compute_units=NE)
ma  = ct.models.MLModel(f"{GE}/edgetam_memory_attention.mlpackage",  compute_units=GPU)
dec = ct.models.MLModel(f"{GE}/edgetam_mask_decoder_nhwc.mlpackage",  compute_units=DEC_UNIT)

def do_enc():  return enc.predict({"image_norm": img})
def do_ma(curr): return ma.predict({"curr": curr, "memory": memory, "curr_pos": curr_pos, "memory_pos": memory_pos})
def do_dec(cond, n0, n1):
    return dec.predict({"image_embeddings": cond, "image_pe": image_pe, "sparse": sparse,
                        "dense": dense, "feat_s0": n0, "feat_s1": n1})

# warm/compile every model on its unit before any measurement
for _ in range(10):
    e = do_enc(); n0, n1, n2 = a(e["neck0"]), a(e["neck1"]), a(e["neck2"])
    m = do_ma(n2.reshape(4096, 1, 256))
    do_dec(a(m["var_304"]).reshape(1, 64, 64, 256), n0, n1)

# ---------------------------------------------------------------------------
# PROBE: do two coremltools predicts on DIFFERENT units overlap, or serialize?
# Run encoder(ANE) K times and mem-attn(GPU) K times, first back-to-back in one
# thread (=sum), then concurrently in two threads. overlap = sum / wall_concurrent.
# ~2.0 => GIL released, true cross-unit concurrency. ~1.0 => serialized (GIL held).
# ---------------------------------------------------------------------------
K = 40
n2_fixed = a(do_enc()["neck2"]).reshape(4096, 1, 256)
t0 = time.perf_counter(); [do_enc() for _ in range(K)]; t_enc = time.perf_counter() - t0
t0 = time.perf_counter(); [do_ma(n2_fixed) for _ in range(K)]; t_ma = time.perf_counter() - t0
def loop_enc():
    for _ in range(K): do_enc()
def loop_ma():
    for _ in range(K): do_ma(n2_fixed)
ta, tb = threading.Thread(target=loop_enc), threading.Thread(target=loop_ma)
t0 = time.perf_counter(); ta.start(); tb.start(); ta.join(); tb.join()
t_both = time.perf_counter() - t0
overlap = (t_enc + t_ma) / t_both
print(f"\n[GIL/overlap probe]  enc {t_enc*1000/K:.1f}ms  ma {t_ma*1000/K:.1f}ms  "
      f"serial-sum {(t_enc+t_ma)*1000/K:.1f}ms  concurrent {t_both*1000/K:.1f}ms")
print(f"[GIL/overlap probe]  overlap factor = {overlap:.2f}x  "
      f"({'TRUE concurrency (GIL released)' if overlap > 1.4 else 'SERIALIZED (GIL held — threading wont help in Python)'})")

# ---------------------------------------------------------------------------
# 3-stage pipeline: 3 worker threads, 2 bounded queues (maxsize=2).
# ---------------------------------------------------------------------------
N = 100
q1 = queue.Queue(maxsize=2)   # encoder -> mem-attn  (carries curr + neck0/neck1 for the decoder)
q2 = queue.Queue(maxsize=2)   # mem-attn -> decoder
enq_t = [0.0] * N             # when frame entered stage 1
done_t = [0.0] * N            # when frame left stage 3

def stage_encoder():
    for i in range(N):
        enq_t[i] = time.perf_counter()
        e = do_enc()
        n0, n1, n2 = a(e["neck0"]), a(e["neck1"]), a(e["neck2"])
        q1.put((i, n2.reshape(4096, 1, 256), n0, n1))
    q1.put(None)

def stage_memattn():
    while True:
        item = q1.get()
        if item is None:
            q2.put(None); break
        i, curr, n0, n1 = item
        m = do_ma(curr)
        q2.put((i, a(m["var_304"]).reshape(1, 64, 64, 256), n0, n1))

def stage_decoder():
    while True:
        item = q2.get()
        if item is None:
            break
        i, cond, n0, n1 = item
        do_dec(cond, n0, n1)
        done_t[i] = time.perf_counter()

threads = [threading.Thread(target=f) for f in (stage_encoder, stage_memattn, stage_decoder)]
T0 = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
wall = time.perf_counter() - T0

lat = sorted((done_t[i] - enq_t[i]) * 1000 for i in range(N))
fps = N / wall
print(f"\n3-stage THREADED pipeline ({N} frames, dec={os.environ.get('SAM3_DEC_UNIT','ane')}):")
print(f"  wall:        {wall*1000:7.1f} ms  =>  {fps:.1f} fps")
print(f"  per-frame latency  p50 {st.median(lat):.1f} ms   p95 {lat[int(0.95*N)]:.1f} ms")
print(f"  vs sequential 3-stage (pipeline_coreml.py): 19.1 fps")
print(f"  speedup over sequential: {fps/19.1:.2f}x")
