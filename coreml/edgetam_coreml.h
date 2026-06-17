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
// (already ImageNet-normalized, same NCHW layout as the ggml path's
// preprocessing). The model exports neck(trunk(x))[0:3] — the 256-ch fused FPN
// levels that match ggml's neck_trk — CHANNELS-LAST [1,H,W,D], whose contiguous
// bytes equal ggml's neck_trk [D,W,H]. Outputs (f32, widened from the model's
// FP16) per level: neck0 256×256×256, neck1 128×128×256, neck2 64×64×256.
// Caller pre-allocates all three (D*W*H floats each). Returns 1 on success.
int edgetam_coreml_encode(edgetam_coreml_handle h, const float* input_norm,
                          float* neck0, float* neck1, float* neck2);

// Run the memory-attention model (a separate handle/model from the encoder).
// All inputs are f32 contiguous; ggml's [feature,token] layout is byte-identical
// to the model's [token,1,feature], so the caller passes ggml buffers directly:
//   curr,curr_pos: 4096*256 floats; memory,memory_pos: 3648*64 floats.
// `conditioned` (4096*256 floats, widened from the model's FP16) receives the
// attended output (== ggml mem-attn output layout). Returns 1 on success.
// Only valid at the fixed steady-state capacity (7 memory frames + 16 obj-ptrs).
int edgetam_coreml_memattn(edgetam_coreml_handle h,
                           const float* curr, const float* memory,
                           const float* curr_pos, const float* memory_pos,
                           float* conditioned);

// Run the SAM mask decoder (a separate handle/model). All inputs are f32
// CHANNELS-LAST [1,H,W,C], byte-identical to ggml's channels-inner buffers, so
// the caller passes them directly:
//   image_embeddings,image_pe,dense: 256*64*64 ; sparse: 256 ;
//   feat_s0: 256*256*256 ; feat_s1: 256*128*128.
// Outputs (caller pre-allocates): masks 4*256*256 (4 low-res mask logits),
//   iou_pred 4, obj_score 1, mask_tokens 4*256 (per-mask SAM tokens). The caller
//   does the multimask selection (best of 4 by IoU). Returns 1 on success.
int edgetam_coreml_decode(edgetam_coreml_handle h,
                          const float* image_embeddings, const float* image_pe,
                          const float* sparse, const float* dense,
                          const float* feat_s0, const float* feat_s1,
                          float* masks, float* iou_pred, float* obj_score, float* mask_tokens);

// Run the memory-encode model (memory_encoder + spatial_perceiver; a separate
// handle/model). Inputs f32 contiguous: pix_feat [1,256,64,64] (the encoder's
// top-level 64x64 feature, BCHW) and mask_logits [1,1,1024,1024] (the decoder's
// full-res mask). Outputs (caller pre-allocates 512*64 floats each): mem_feats
// the 512 compressed memory latents [1,512,64], mem_pos their position encoding
// [1,512,64] — exactly what one new slot of the mem-attn memory bank consumes.
// Exported with parity cosine 1.000000; ~5 ms on ANE/GPU. Returns 1 on success.
int edgetam_coreml_memencode(edgetam_coreml_handle h,
                             const float* pix_feat, const float* mask_logits,
                             float* mem_feats, float* mem_pos);

void edgetam_coreml_destroy(edgetam_coreml_handle h);

#ifdef __cplusplus
}
#endif

#endif  // EDGETAM_COREML_H
