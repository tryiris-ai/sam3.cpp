// RFD 0011 U8 — implementation of the extern "C" tracker C-ABI (edgetam_capi.h).
// Thin POD wrapper over the sam3.cpp visual tracker (sam3_create_visual_tracker /
// sam3_encode_image / sam3_tracker_add_instance / sam3_propagate_frame). All C++
// exceptions are caught at the boundary (cgo cannot unwind through Go).
#include "edgetam_capi.h"
#include "../sam3.h"
#include "edgetam_coreml.h"   // producer encoder handle (edgetam_coreml_create/encode/destroy)

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct EtTracker {
    std::shared_ptr<sam3_model> model;
    sam3_state_ptr              state;
    sam3_tracker_ptr            tracker;

    // RFD 0011 U8 Wave 5 (encoder-ahead threading): a producer encoder handle
    // (its OWN MLModel, per-thread ownership) + a fixed neck pool. Go owns slot
    // free/ready coordination via channels, so NO lock here — each slot is written
    // once by encode_slot (producer thread) then read once by track_slot (consumer
    // thread). track_slot is the ONLY caller of sam3_coreml_set_prefetched_neck
    // (global, consumer-thread-only). prod_enc is non-null only when CoreML is on.
    edgetam_coreml_handle prod_enc = nullptr;
    static const int      POOL = 3;
    std::vector<float>    neck[POOL][3];   // n0 256*256*256, n1 128*128*256, n2 64*64*256

    // RFD 0011 U8 mask export: the last track's binary mask (0/255, orig-frame
    // resolution), stashed so the Go side can pull the pixels after a track()
    // for State-Sync/frontend rendering. Cleared on a lost frame.
    sam3_mask last_mask;
};

// Wrap a caller RGB24 buffer as a sam3_image (copies into the owned vector — the
// library keeps no alias, and cgo must not let Go memory escape into C).
sam3_image make_image(const uint8_t* rgb, int w, int h) {
    sam3_image img;
    img.width = w; img.height = h; img.channels = 3;
    img.data.assign(rgb, rgb + (size_t)w * h * 3);
    return img;
}

}  // namespace

extern "C" edgetam_tracker_t edgetam_capi_create(const char* models_dir,
                                                 const char* ggml_model,
                                                 int /*use_samurai*/, int /*use_lifecycle*/) {
    try {
#ifdef SAM3_COREML
        // Enable the CoreML stages (encoder ANE + mem-attn GPU + decoder ANE) by
        // pointing the library's existing runtime gates at the .mlpackage dir.
        // The library's full-CoreML propagation path then runs with zero ggml
        // graph work per frame; ggml remains the seed + early-frame fallback.
        // On a non-CoreML build (e.g. Windows/Vulkan, SAM3_VULKAN) this block is
        // absent → models_dir is ignored and the tracker runs the pure-ggml path,
        // which lands on whatever ggml GPU backend is compiled (Vulkan).
        if (models_dir && *models_dir) {
            std::string d = models_dir;
            setenv("SAM3_COREML_ENCODER", "1", 1);
            setenv("SAM3_COREML_MODEL", (d + "/edgetam_encoder_neck_nhwc.mlpackage").c_str(), 1);
            setenv("SAM3_COREML_MEMATTN", "1", 1);
            setenv("SAM3_COREML_MEMATTN_MODEL", (d + "/edgetam_memory_attention.mlpackage").c_str(), 1);
            setenv("SAM3_COREML_DECODER", "1", 1);
            setenv("SAM3_COREML_DECODER_MODEL", (d + "/edgetam_mask_decoder_nhwc.mlpackage").c_str(), 1);
            // Memory-encode on ANE + a fixed-size (padded) memory bank. The CoreML
            // mem-attention model has a static 3648-token input (7*512 + 16*4); PAD_BANK
            // pads the live bank to that shape so MEMATTN actually runs on the ANE/GPU
            // instead of silently falling back to the ~184ms ggml path. Without these two
            // the tracker is correct but ~8x slower (measured 570ms vs 69ms/frame).
            setenv("SAM3_COREML_MEMENC", "1", 1);
            setenv("SAM3_COREML_MEMENC_MODEL", (d + "/edgetam_memory_encode.mlpackage").c_str(), 1);
            setenv("SAM3_COREML_PAD_BANK", "1", 1);
        }
#else
        (void)models_dir;  // Vulkan/CPU build: pure-ggml path, no CoreML stage gates
#endif
        auto t = std::make_unique<EtTracker>();
        sam3_params p;
        p.model_path = ggml_model ? ggml_model : "";
        p.use_gpu = true;
        t->model = sam3_load_model(p);
        if (!t->model) return nullptr;
        t->state = sam3_create_state(*t->model, p);
        sam3_visual_track_params vp;  // defaults match sam3_edgetam_bench
        t->tracker = sam3_create_visual_tracker(*t->model, vp);
#ifdef SAM3_COREML
        // Wave 5: producer encoder handle + neck pool (CoreML encoder-ahead threading).
        // Absent on non-CoreML builds → prod_enc stays null → pool_size()==0 →
        // the cgo session uses the synchronous (single-thread) Track path.
        if (models_dir && *models_dir) {
            std::string d = models_dir;
            t->prod_enc = edgetam_coreml_create((d + "/edgetam_encoder_neck_nhwc.mlpackage").c_str(), /*ANE*/ 1);
            const int Wd[3] = {256, 128, 64};
            for (int s = 0; s < EtTracker::POOL; ++s)
                for (int i = 0; i < 3; ++i)
                    t->neck[s][i].assign((size_t)256 * Wd[i] * Wd[i], 0.f);
        }
#endif
        return t.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" int edgetam_capi_seed(edgetam_tracker_t h, const uint8_t* rgb, int w, int hgt, et_box seed) {
    auto* t = static_cast<EtTracker*>(h);
    if (!t || !rgb) return 0;
    try {
        sam3_image img = make_image(rgb, w, hgt);
        if (!sam3_encode_image(*t->state, *t->model, img)) return 0;
        sam3_pvs_params pvs;
        pvs.use_box = true;
        pvs.box = {seed.x0, seed.y0, seed.x1, seed.y1};
        int id = sam3_tracker_add_instance(*t->tracker, *t->state, *t->model, pvs);
        return id >= 0 ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

extern "C" et_result edgetam_capi_track(edgetam_tracker_t h, const uint8_t* rgb, int w, int hgt) {
    et_result r;
    std::memset(&r, 0, sizeof(r));
    auto* t = static_cast<EtTracker*>(h);
    if (!t || !rgb) return r;
    try {
        sam3_image img = make_image(rgb, w, hgt);
        sam3_result res = sam3_propagate_frame(*t->tracker, *t->state, *t->model, img);
        if (res.detections.empty()) { t->last_mask = sam3_mask{}; return r; }  // lost -> clear mask
        const sam3_detection& d = res.detections[0];
        t->last_mask = d.mask;  // stash for edgetam_capi_last_mask
        const float W = (float)w, H = (float)hgt;
        r.box.x0 = d.box.x0 / W; r.box.y0 = d.box.y0 / H;
        r.box.x1 = d.box.x1 / W; r.box.y1 = d.box.y1 / H;
        r.valid     = 1;
        r.obj_score = d.score;
        r.mask_iou  = d.iou_score;
        r.state     = (int)d.state;
        return r;
    } catch (...) {
        return r;
    }
}

// Copies the last track's binary mask (0/255, row-major, w*h bytes) into `out`
// (capacity `cap`). Writes dims to *w,*h. Returns the mask byte count (w*h); if
// cap < w*h it copies nothing and returns w*h so the caller can size the buffer.
// 0 => no mask this frame (lost / not yet tracked). RFD 0011 U8 mask export.
extern "C" int edgetam_capi_last_mask(edgetam_tracker_t h, uint8_t* out, int cap, int* w, int* hgt) {
    auto* t = static_cast<EtTracker*>(h);
    if (!t) return 0;
    const sam3_mask& m = t->last_mask;
    const int n = (int)m.data.size();
    if (w) *w = m.width;
    if (hgt) *hgt = m.height;
    if (n == 0) return 0;
    if (out && cap >= n) std::memcpy(out, m.data.data(), (size_t)n);
    return n;
}

extern "C" void edgetam_capi_reset(edgetam_tracker_t h) {
    auto* t = static_cast<EtTracker*>(h);
    if (t && t->tracker) {
        try { sam3_tracker_reset(*t->tracker); } catch (...) {}
    }
}

extern "C" void edgetam_capi_destroy(edgetam_tracker_t h) {
    auto* t = static_cast<EtTracker*>(h);
#ifdef SAM3_COREML
    if (t && t->prod_enc) edgetam_coreml_destroy(t->prod_enc);
#endif
    delete t;
}

// ── RFD 0011 U8 Wave 5: encoder-ahead split ──────────────────────────────────
// Go drives the pipeline: a producer goroutine pops a free slot, calls
// encode_slot; a consumer goroutine calls track_slot then recycles the slot. The
// two run on different OS threads (per-thread MLModel ownership): encode_slot uses
// the producer encoder handle; track_slot consumes the prefetched neck and never
// encodes. Go's channels serialize each slot (write-then-read), so no lock here.

extern "C" int edgetam_capi_pool_size(edgetam_tracker_t h) {
    auto* t = static_cast<EtTracker*>(h);
    return (t && t->prod_enc) ? EtTracker::POOL : 0;   // 0 => threading unavailable
}

extern "C" int edgetam_capi_encode_slot(edgetam_tracker_t h, int slot,
                                        const uint8_t* rgb, int w, int hgt) {
    auto* t = static_cast<EtTracker*>(h);
    if (!t || !t->prod_enc || !rgb || slot < 0 || slot >= EtTracker::POOL) return 0;
#ifdef SAM3_COREML
    try {
        sam3_image img = make_image(rgb, w, hgt);
        std::vector<float> norm = sam3_coreml_preprocess_image(img, 1024);
        return edgetam_coreml_encode(t->prod_enc, norm.data(),
                                     t->neck[slot][0].data(),
                                     t->neck[slot][1].data(),
                                     t->neck[slot][2].data()) ? 1 : 0;
    } catch (...) {
        return 0;
    }
#else
    (void)rgb; (void)w; (void)hgt;  // encoder-ahead threading is CoreML-only
    return 0;
#endif
}

extern "C" et_result edgetam_capi_track_slot(edgetam_tracker_t h, int slot,
                                             const uint8_t* rgb, int w, int hgt) {
    et_result r;
    std::memset(&r, 0, sizeof(r));
    auto* t = static_cast<EtTracker*>(h);
    if (!t || !rgb || slot < 0 || slot >= EtTracker::POOL) return r;
    try {
#ifdef SAM3_COREML
        // Consumer thread only: hand the producer-encoded neck to the library so
        // sam3_propagate_frame's encode step is skipped (no ggml/CoreML encode here).
        sam3_coreml_set_prefetched_neck(t->neck[slot][0].data(),
                                        t->neck[slot][1].data(),
                                        t->neck[slot][2].data());
#endif
        sam3_image img = make_image(rgb, w, hgt);
        sam3_result res = sam3_propagate_frame(*t->tracker, *t->state, *t->model, img);
        if (res.detections.empty()) { t->last_mask = sam3_mask{}; return r; }
        const sam3_detection& d = res.detections[0];
        t->last_mask = d.mask;  // stash for edgetam_capi_last_mask
        const float W = (float)w, H = (float)hgt;
        r.box.x0 = d.box.x0 / W; r.box.y0 = d.box.y0 / H;
        r.box.x1 = d.box.x1 / W; r.box.y1 = d.box.y1 / H;
        r.valid     = 1;
        r.obj_score = d.score;
        r.mask_iou  = d.iou_score;
        r.state     = (int)d.state;
        return r;
    } catch (...) {
        return r;
    }
}

extern "C" const char* edgetam_capi_version(void) {
    return SAM3_VERSION;
}

extern "C" const char* edgetam_capi_backend(edgetam_tracker_t h) {
    auto* t = static_cast<EtTracker*>(h);
    // *t->model matches the existing capi pattern (sam3_encode_image(*t->state,
    // *t->model, ...)); sam3_backend_name reads the model's active ggml backend.
    return (t && t->model) ? sam3_backend_name(*t->model) : "none";
}
