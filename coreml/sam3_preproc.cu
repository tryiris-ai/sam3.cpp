// sam3_preproc.cu — device frame preprocess (Windows parity plan v1 STEP 5).
//
// Replicates sam3.cpp's CPU reference BIT-EXACTLY:
//   sam3_resize_bilinear (torch bilinear align_corners=False, ALL arithmetic
//   in double, round-to-nearest to uint8)  →  sam2_preprocess_image's
//   ImageNet normalize in float, CHW.
// Bit-exactness contract: double precision per output pixel in the SAME
// operation order as the CPU loop; this file is compiled with -fmad=false so
// no FMA contraction can perturb the double chain (IEEE round-to-nearest per
// op on both MSVC x64 and sm_86/89 then yields identical bits); the float
// normalize is two float ops in the same order. Verified at runtime by
// SAM3_PREPROC_DEVICE=2 (compute both paths, memcmp, abort on mismatch).
//
// The kernel writes the f32 CHW result DIRECTLY into the encoder input
// tensor's device memory (ggml-CUDA tensor->data is a raw device pointer),
// replacing a 12.6 MB f32 H2D with a ~2.7 MB u8 upload and deleting the
// ~14ms single-threaded CPU preprocess from the hold critical path.

#include <cuda_runtime.h>
#include <stdint.h>

static __global__ void sam3_preproc_kernel(const uint8_t* __restrict__ src,
                                           int src_w, int src_h,
                                           float* __restrict__ dst,
                                           int img_size,
                                           float mean0, float mean1, float mean2,
                                           float std0, float std1, float std2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_size || y >= img_size) return;

    const double sx = (double)src_w / img_size;
    const double sy = (double)src_h / img_size;

    double fy = (y + 0.5) * sy - 0.5;
    if (fy < 0.0) fy = 0.0;
    const int y0 = (int)fy;
    const int y1 = (y0 < src_h - 1) ? y0 + 1 : y0;
    const double wy = fy - y0;
    const double wy0 = 1.0 - wy;

    double fx = (x + 0.5) * sx - 0.5;
    if (fx < 0.0) fx = 0.0;
    const int x0 = (int)fx;
    const int x1 = (x0 < src_w - 1) ? x0 + 1 : x0;
    const double wx = fx - x0;
    const double wx0 = 1.0 - wx;

    const float mean[3] = {mean0, mean1, mean2};
    const float std_d[3] = {std0, std1, std2};
    const int plane = img_size * img_size;

    for (int c = 0; c < 3; ++c) {
        const double p00 = src[(y0 * src_w + x0) * 3 + c];
        const double p01 = src[(y0 * src_w + x1) * 3 + c];
        const double p10 = src[(y1 * src_w + x0) * 3 + c];
        const double p11 = src[(y1 * src_w + x1) * 3 + c];
        // Same association order as the CPU reference.
        double v = wy0 * (wx0 * p00 + wx * p01) +
                   wy * (wx0 * p10 + wx * p11);
        int iv = (int)(v + 0.5);
        if (iv < 0) iv = 0;
        if (iv > 255) iv = 255;
        // CPU reference: float division by 255.0f, then (v-mean)/std in float.
        float fv = (float)(uint8_t)iv / 255.0f;
        dst[c * plane + y * img_size + x] = (fv - mean[c]) / std_d[c];
    }
}

// Identity fast path (src already img_size×img_size): normalize only,
// matching the CPU reference's no-resize branch.
static __global__ void sam3_preproc_norm_kernel(const uint8_t* __restrict__ src,
                                                float* __restrict__ dst,
                                                int img_size,
                                                float mean0, float mean1, float mean2,
                                                float std0, float std1, float std2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_size || y >= img_size) return;
    const float mean[3] = {mean0, mean1, mean2};
    const float std_d[3] = {std0, std1, std2};
    const int plane = img_size * img_size;
    for (int c = 0; c < 3; ++c) {
        float fv = (float)src[(y * img_size + x) * 3 + c] / 255.0f;
        dst[c * plane + y * img_size + x] = (fv - mean[c]) / std_d[c];
    }
}

// C entry point (called from sam3.cpp under GGML_USE_CUDA + device
// transport). dst_dev = the encoder input tensor's device pointer (f32 CHW,
// img_size²·3). Uses an owned staging buffer + stream; host-synchronizes
// before returning so the subsequent graph launch (any stream) observes the
// writes. Returns false on any CUDA error — caller falls back to the CPU
// path (fail-soft: preprocess must never kill a frame).
extern "C" bool sam3_cuda_preprocess_imagenet(const uint8_t* rgb, int src_w, int src_h,
                                              float* dst_dev, int img_size) {
    static uint8_t* s_stage = nullptr;
    static size_t s_stage_cap = 0;
    const size_t need = (size_t)src_w * src_h * 3;
    if (need > s_stage_cap) {
        if (s_stage) cudaFree(s_stage);
        if (cudaMalloc(&s_stage, need) != cudaSuccess) { s_stage = nullptr; s_stage_cap = 0; return false; }
        s_stage_cap = need;
    }
    if (cudaMemcpy(s_stage, rgb, need, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    dim3 blk(16, 16);
    dim3 grd((img_size + 15) / 16, (img_size + 15) / 16);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_d[3] = {0.229f, 0.224f, 0.225f};
    if (src_w == img_size && src_h == img_size) {
        sam3_preproc_norm_kernel<<<grd, blk>>>(s_stage, dst_dev, img_size,
                                               mean[0], mean[1], mean[2],
                                               std_d[0], std_d[1], std_d[2]);
    } else {
        sam3_preproc_kernel<<<grd, blk>>>(s_stage, src_w, src_h, dst_dev, img_size,
                                          mean[0], mean[1], mean[2],
                                          std_d[0], std_d[1], std_d[2]);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    // Host sync: the caller launches the encoder graph after this returns;
    // host ordering guarantees visibility across streams.
    return cudaDeviceSynchronize() == cudaSuccess;
}
