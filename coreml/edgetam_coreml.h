// RFD 0011 (CoreML/ANE hybrid) — C ABI for running the EdgeTAM image encoder
// via CoreML (Apple Neural Engine) instead of the ggml/Metal path. Measured at
// ~12 ms on the ANE vs ~122 ms on ggml/Metal (~10x). The C++ tracker keeps
// doing its existing ImageNet preprocessing and hands the normalized float
// tensor here; outputs come back as PyTorch-NCHW fp32 (the caller maps them
// into the ggml neck_trk layout).
//
// Implemented in edgetam_coreml.mm (Objective-C++, links CoreML + Foundation).
// On non-Apple builds this header is unused (guarded by SAM3_COREML in CMake).
#ifndef EDGETAM_COREML_H
#define EDGETAM_COREML_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a loaded CoreML encoder.
typedef void* edgetam_coreml_handle;

// Load (and compile, if a .mlpackage) the encoder. compute_units:
//   0 = ALL, 1 = CPU_AND_NE (ANE), 2 = CPU_AND_GPU, 3 = CPU_ONLY.
// Returns NULL on failure.
edgetam_coreml_handle edgetam_coreml_create(const char* model_path, int compute_units);

// Run the encoder. `input_norm` is a contiguous [1,3,1024,1024] f32 buffer
// (already ImageNet-normalized, same as the ggml path's preprocessing).
// Outputs are written as contiguous PyTorch-NCHW f32:
//   vision_features [1,256,64,64], hr0 [1,32,256,256], hr1 [1,64,128,128].
// Caller pre-allocates all three. Returns 1 on success, 0 on failure.
int edgetam_coreml_encode(edgetam_coreml_handle h, const float* input_norm,
                          float* vision_features, float* hr0, float* hr1);

void edgetam_coreml_destroy(edgetam_coreml_handle h);

#ifdef __cplusplus
}
#endif

#endif  // EDGETAM_COREML_H
