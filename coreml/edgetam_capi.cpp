// RFD 0011 U8 — implementation of the extern "C" tracker C-ABI (edgetam_capi.h).
// Thin POD wrapper over the sam3.cpp visual tracker (sam3_create_visual_tracker /
// sam3_encode_image / sam3_tracker_add_instance / sam3_propagate_frame). All C++
// exceptions are caught at the boundary (cgo cannot unwind through Go).
#include "edgetam_capi.h"
#include "../sam3.h"

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
        // Enable the CoreML stages (encoder ANE + mem-attn GPU + decoder ANE) by
        // pointing the library's existing runtime gates at the .mlpackage dir.
        // The library's full-CoreML propagation path then runs with zero ggml
        // graph work per frame; ggml remains the seed + early-frame fallback.
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
        auto t = std::make_unique<EtTracker>();
        sam3_params p;
        p.model_path = ggml_model ? ggml_model : "";
        p.use_gpu = true;
        t->model = sam3_load_model(p);
        if (!t->model) return nullptr;
        t->state = sam3_create_state(*t->model, p);
        sam3_visual_track_params vp;  // defaults match sam3_edgetam_bench
        t->tracker = sam3_create_visual_tracker(*t->model, vp);
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
        if (res.detections.empty()) return r;  // valid stays 0 -> lost
        const sam3_detection& d = res.detections[0];
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

extern "C" void edgetam_capi_reset(edgetam_tracker_t h) {
    auto* t = static_cast<EtTracker*>(h);
    if (t && t->tracker) {
        try { sam3_tracker_reset(*t->tracker); } catch (...) {}
    }
}

extern "C" void edgetam_capi_destroy(edgetam_tracker_t h) {
    delete static_cast<EtTracker*>(h);
}

extern "C" const char* edgetam_capi_version(void) {
    return SAM3_VERSION;
}
