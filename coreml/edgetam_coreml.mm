// RFD 0011 (CoreML/ANE hybrid) — Objective-C++ implementation of the EdgeTAM
// CoreML encoder bridge. See edgetam_coreml.h. Links CoreML + Foundation.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "edgetam_coreml.h"
#include <cstring>
#include <cstdio>

namespace {
// ARC-managed holder (a C++ struct with a strong ObjC member is managed by ARC
// in an ObjC++ .mm translation unit).
struct EdgetamCoreML {
    MLModel* model = nil;
};

// Copy an MLMultiArray's backing bytes into `dst` (pre-sized to `count` floats).
// An FP16 CoreML model emits Float16 outputs, so handle both Float32 (memcpy)
// and Float16 (widen each element). getBytesWithHandler hands back the
// contiguous C-order backing buffer.
static bool copy_f32(MLMultiArray* arr, float* dst, NSInteger count) {
    if (!arr) return false;
    __block bool ok = false;
    if (arr.dataType == MLMultiArrayDataTypeFloat32) {
        [arr getBytesWithHandler:^(const void* bytes, NSInteger size) {
            NSInteger want = count * (NSInteger)sizeof(float);
            if (size >= want) { memcpy(dst, bytes, want); ok = true; }
            else NSLog(@"[edgetam_coreml] f32 output too small: %ld < %ld", (long)size, (long)want);
        }];
    } else if (arr.dataType == MLMultiArrayDataTypeFloat16) {
        [arr getBytesWithHandler:^(const void* bytes, NSInteger size) {
            NSInteger want = count * (NSInteger)sizeof(__fp16);
            if (size >= want) {
                const __fp16* src = (const __fp16*)bytes;     // ARM64 half precision
                for (NSInteger i = 0; i < count; ++i) dst[i] = (float)src[i];
                ok = true;
            } else NSLog(@"[edgetam_coreml] f16 output too small: %ld < %ld", (long)size, (long)want);
        }];
    } else {
        NSLog(@"[edgetam_coreml] unexpected output dtype %ld", (long)arr.dataType);
    }
    return ok;
}
}  // namespace

extern "C" {

edgetam_coreml_handle edgetam_coreml_create(const char* model_path, int compute_units) {
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:model_path];
        NSURL* url = [NSURL fileURLWithPath:path];
        NSError* err = nil;

        // .mlpackage / .mlmodel must be compiled to a .mlmodelc first.
        NSURL* compiled = url;
        if ([path hasSuffix:@".mlpackage"] || [path hasSuffix:@".mlmodel"]) {
            compiled = [MLModel compileModelAtURL:url error:&err];
            if (err || !compiled) {
                NSLog(@"[edgetam_coreml] compile failed: %@", err);
                return nullptr;
            }
        }

        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        switch (compute_units) {
            case 1:  cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine; break;
            case 2:  cfg.computeUnits = MLComputeUnitsCPUAndGPU;          break;
            case 3:  cfg.computeUnits = MLComputeUnitsCPUOnly;            break;
            default: cfg.computeUnits = MLComputeUnitsAll;                break;
        }

        MLModel* m = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
        if (err || !m) {
            NSLog(@"[edgetam_coreml] load failed: %@", err);
            return nullptr;
        }
        auto* h = new EdgetamCoreML();
        h->model = m;
        return (edgetam_coreml_handle)h;
    }
}

int edgetam_coreml_encode(edgetam_coreml_handle handle, const float* input_norm,
                          float* neck0, float* neck1, float* neck2) {
    @autoreleasepool {
        auto* h = (EdgetamCoreML*)handle;
        if (!h || !h->model) return 0;
        NSError* err = nil;

        MLMultiArray* in = [[MLMultiArray alloc] initWithShape:@[@1, @3, @1024, @1024]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:&err];
        if (err || !in) { NSLog(@"[edgetam_coreml] input alloc: %@", err); return 0; }
        memcpy(in.dataPointer, input_norm, sizeof(float) * 1 * 3 * 1024 * 1024);

        MLDictionaryFeatureProvider* fp =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"image_norm": [MLFeatureValue featureValueWithMultiArray:in]}
                             error:&err];
        if (err || !fp) { NSLog(@"[edgetam_coreml] feature provider: %@", err); return 0; }

        id<MLFeatureProvider> out = [h->model predictionFromFeatures:fp error:&err];
        if (err || !out) { NSLog(@"[edgetam_coreml] predict failed: %@", err); return 0; }

        // The encoder exports neck(trunk(x))[0:3] — the 256-ch fused FPN levels
        // that match ggml's neck_trk. neck0 256x256, neck1 128x128, neck2 64x64.
        MLMultiArray* n0  = [[out featureValueForName:@"neck0"] multiArrayValue];
        MLMultiArray* n1  = [[out featureValueForName:@"neck1"] multiArrayValue];
        MLMultiArray* n2  = [[out featureValueForName:@"neck2"] multiArrayValue];

        bool ok = copy_f32(n0, neck0, 1 * 256 * 256 * 256)
               && copy_f32(n1, neck1, 1 * 256 * 128 * 128)
               && copy_f32(n2, neck2, 1 * 256 * 64 * 64);
        return ok ? 1 : 0;
    }
}

int edgetam_coreml_memattn(edgetam_coreml_handle handle,
                           const float* curr, const float* memory,
                           const float* curr_pos, const float* memory_pos,
                           float* conditioned) {
    @autoreleasepool {
        auto* h = (EdgetamCoreML*)handle;
        if (!h || !h->model) return 0;
        NSError* err = nil;
        // ggml [feature,token] == model [token,1,feature] byte-for-byte (feature
        // innermost), so each input is a direct memcpy into the MLMultiArray.
        auto mk = [&](const float* data, int ntok, int feat) -> MLMultiArray* {
            MLMultiArray* a = [[MLMultiArray alloc] initWithShape:@[@(ntok), @1, @(feat)]
                                                         dataType:MLMultiArrayDataTypeFloat32
                                                            error:&err];
            if (a) memcpy(a.dataPointer, data, sizeof(float) * (size_t)ntok * feat);
            return a;
        };
        MLMultiArray* c  = mk(curr,       4096, 256);
        MLMultiArray* m  = mk(memory,     3648, 64);
        MLMultiArray* cp = mk(curr_pos,   4096, 256);
        MLMultiArray* mp = mk(memory_pos, 3648, 64);
        if (!c || !m || !cp || !mp) { NSLog(@"[edgetam_coreml] memattn input alloc: %@", err); return 0; }

        MLDictionaryFeatureProvider* fp =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"curr":       [MLFeatureValue featureValueWithMultiArray:c],
                                     @"memory":     [MLFeatureValue featureValueWithMultiArray:m],
                                     @"curr_pos":   [MLFeatureValue featureValueWithMultiArray:cp],
                                     @"memory_pos": [MLFeatureValue featureValueWithMultiArray:mp]}
                             error:&err];
        if (err || !fp) { NSLog(@"[edgetam_coreml] memattn feature provider: %@", err); return 0; }

        id<MLFeatureProvider> out = [h->model predictionFromFeatures:fp error:&err];
        if (err || !out) { NSLog(@"[edgetam_coreml] memattn predict: %@", err); return 0; }
        MLMultiArray* o = [[out featureValueForName:@"var_304"] multiArrayValue];
        return copy_f32(o, conditioned, 1 * 4096 * 256) ? 1 : 0;
    }
}

int edgetam_coreml_decode(edgetam_coreml_handle handle,
                          const float* image_embeddings, const float* image_pe,
                          const float* sparse, const float* dense,
                          const float* feat_s0, const float* feat_s1,
                          float* masks, float* iou_pred, float* obj_score, float* mask_tokens) {
    @autoreleasepool {
        auto* h = (EdgetamCoreML*)handle;
        if (!h || !h->model) return 0;
        NSError* err = nil;
        // Channels-last [1,H,W,C] inputs == ggml channels-inner buffers (memcpy).
        auto mk = [&](const float* data, NSArray<NSNumber*>* shape, size_t n) -> MLMultiArray* {
            MLMultiArray* a = [[MLMultiArray alloc] initWithShape:shape
                                                         dataType:MLMultiArrayDataTypeFloat32 error:&err];
            if (a) memcpy(a.dataPointer, data, sizeof(float) * n);
            return a;
        };
        MLMultiArray* ie = mk(image_embeddings, @[@1,@64,@64,@256],   (size_t)256*64*64);
        MLMultiArray* pe = mk(image_pe,         @[@1,@64,@64,@256],   (size_t)256*64*64);
        MLMultiArray* sp = mk(sparse,           @[@1,@1,@256],        (size_t)256);
        MLMultiArray* de = mk(dense,            @[@1,@64,@64,@256],   (size_t)256*64*64);
        MLMultiArray* f0 = mk(feat_s0,          @[@1,@256,@256,@256], (size_t)256*256*256);
        MLMultiArray* f1 = mk(feat_s1,          @[@1,@128,@128,@256], (size_t)256*128*128);
        if (!ie||!pe||!sp||!de||!f0||!f1) { NSLog(@"[edgetam_coreml] decode input alloc: %@", err); return 0; }

        MLDictionaryFeatureProvider* fp =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"image_embeddings":[MLFeatureValue featureValueWithMultiArray:ie],
                                     @"image_pe":        [MLFeatureValue featureValueWithMultiArray:pe],
                                     @"sparse":          [MLFeatureValue featureValueWithMultiArray:sp],
                                     @"dense":           [MLFeatureValue featureValueWithMultiArray:de],
                                     @"feat_s0":         [MLFeatureValue featureValueWithMultiArray:f0],
                                     @"feat_s1":         [MLFeatureValue featureValueWithMultiArray:f1]}
                             error:&err];
        if (err || !fp) { NSLog(@"[edgetam_coreml] decode feature provider: %@", err); return 0; }

        id<MLFeatureProvider> out = [h->model predictionFromFeatures:fp error:&err];
        if (err || !out) { NSLog(@"[edgetam_coreml] decode predict: %@", err); return 0; }
        bool ok = copy_f32([[out featureValueForName:@"masks"]       multiArrayValue], masks,       (size_t)4*256*256)
               && copy_f32([[out featureValueForName:@"iou_pred"]    multiArrayValue], iou_pred,    4)
               && copy_f32([[out featureValueForName:@"obj_score"]   multiArrayValue], obj_score,   1)
               && copy_f32([[out featureValueForName:@"mask_tokens"] multiArrayValue], mask_tokens, (size_t)4*256);
        return ok ? 1 : 0;
    }
}

int edgetam_coreml_memencode(edgetam_coreml_handle handle,
                             const float* pix_feat, const float* mask_logits,
                             float* mem_feats, float* mem_pos) {
    @autoreleasepool {
        auto* h = (EdgetamCoreML*)handle;
        if (!h || !h->model) return 0;
        NSError* err = nil;
        auto mk = [&](const float* data, NSArray<NSNumber*>* shape, size_t n) -> MLMultiArray* {
            MLMultiArray* a = [[MLMultiArray alloc] initWithShape:shape
                                                         dataType:MLMultiArrayDataTypeFloat32 error:&err];
            if (a) memcpy(a.dataPointer, data, sizeof(float) * n);
            return a;
        };
        // pix_feat [1,256,64,64] BCHW, mask_logits [1,1,1024,1024] — same layouts the
        // export traced, so each is a direct memcpy into the MLMultiArray.
        MLMultiArray* pf = mk(pix_feat,    @[@1,@256,@64,@64],   (size_t)256*64*64);
        MLMultiArray* ml = mk(mask_logits, @[@1,@1,@1024,@1024], (size_t)1024*1024);
        if (!pf || !ml) { NSLog(@"[edgetam_coreml] memencode input alloc: %@", err); return 0; }

        MLDictionaryFeatureProvider* fp =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"pix_feat":    [MLFeatureValue featureValueWithMultiArray:pf],
                                     @"mask_logits": [MLFeatureValue featureValueWithMultiArray:ml]}
                             error:&err];
        if (err || !fp) { NSLog(@"[edgetam_coreml] memencode feature provider: %@", err); return 0; }

        id<MLFeatureProvider> out = [h->model predictionFromFeatures:fp error:&err];
        if (err || !out) { NSLog(@"[edgetam_coreml] memencode predict: %@", err); return 0; }
        // Tuple output flattened to var_486 (latents) + var_488 (pos), each [1,512,64].
        bool ok = copy_f32([[out featureValueForName:@"var_486"] multiArrayValue], mem_feats, (size_t)512*64)
               && copy_f32([[out featureValueForName:@"var_488"] multiArrayValue], mem_pos,   (size_t)512*64);
        return ok ? 1 : 0;
    }
}

void edgetam_coreml_destroy(edgetam_coreml_handle handle) {
    auto* h = (EdgetamCoreML*)handle;
    if (h) { h->model = nil; delete h; }
}

}  // extern "C"
