// RFD 0011 U8 — extern "C" C-ABI over the sam3.cpp EdgeTAM visual tracker, for
// cgo into iris/platform. POD only crosses this boundary (cgo cannot call the
// std:: C++ API in sam3.h). The engine underneath is the existing visual tracker
// (memory bank + U3 SAMURAI + U4 lifecycle); CoreML stages are enabled per
// edgetam_capi_create's models_dir. The pure-CoreML 20 fps engine (Wave 1) can
// later swap behind this same ABI without changing the platform cgo side.
#ifndef EDGETAM_CAPI_H
#define EDGETAM_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* edgetam_tracker_t;

// Box corners. seed (edgetam_capi_seed) is in ORIGINAL-FRAME PIXELS; results
// (et_result.box) are NORMALIZED [0,1] (matches hold.Box.Valid on the Go side).
typedef struct { float x0, y0, x1, y1; } et_box;

// state mirrors sam3.h TargetState: 0 TRACKED, 1 AT_RISK, 2 OCCLUDED, 3 LOST,
// 4 CANDIDATE_REACQUIRE, 5 REACQUIRED.
typedef struct {
    et_box box;
    int    valid;        // 1 if a credible detection this frame
    float  obj_score;
    float  mask_iou;
    int    state;
} et_result;

// Create a tracker. ggml_model = path to edgetam_f16.ggml (required, the seed
// path + ggml fallback). models_dir = dir holding the 4 CoreML .mlpackages; if
// non-empty, the CoreML encoder/mem-attn/decoder stages are enabled. Returns
// NULL on failure.
edgetam_tracker_t edgetam_capi_create(const char* models_dir, const char* ggml_model,
                                      int use_samurai, int use_lifecycle);

// Seed the target from a box on frame 0. rgb = contiguous RGB24 (w*h*3 bytes).
// seed box in PIXELS. Returns 1 on success.
int edgetam_capi_seed(edgetam_tracker_t t, const uint8_t* rgb, int w, int h, et_box seed_px);

// Track one frame. rgb = contiguous RGB24 (w*h*3). Returns the per-frame result
// (box normalized [0,1]); valid==0 when the target is lost this frame.
et_result edgetam_capi_track(edgetam_tracker_t t, const uint8_t* rgb, int w, int h);

// RFD 0011 U8 Wave 5 — encoder-ahead threading split. pool_size returns the slot
// count (0 if CoreML/threading is unavailable). encode_slot (PRODUCER thread)
// preprocesses+encodes rgb into slot's neck buffers via the producer's own encoder
// handle. track_slot (CONSUMER thread) consumes slot's prefetched neck + propagates.
// Go owns slot free/ready coordination (channels) — one writer then one reader per
// slot, so these are safe to overlap across the two threads on DIFFERENT slots.
int       edgetam_capi_pool_size(edgetam_tracker_t t);
int       edgetam_capi_encode_slot(edgetam_tracker_t t, int slot, const uint8_t* rgb, int w, int h);
et_result edgetam_capi_track_slot(edgetam_tracker_t t, int slot, const uint8_t* rgb, int w, int h);

void        edgetam_capi_reset(edgetam_tracker_t t);
void        edgetam_capi_destroy(edgetam_tracker_t t);
const char* edgetam_capi_version(void);   // SAM3_VERSION, for cgo link sanity

#ifdef __cplusplus
}
#endif

#endif  // EDGETAM_CAPI_H
