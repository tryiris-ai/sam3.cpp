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

// Copy a float32 MLMultiArray's backing bytes into `dst` (dst pre-sized to
// `count` floats). CoreML model outputs are contiguous C-order; getBytesWithHandler
// hands back the contiguous backing buffer.
static bool copy_f32(MLMultiArray* arr, float* dst, NSInteger count) {
    if (!arr) return false;
    if (arr.dataType != MLMultiArrayDataTypeFloat32) {
        NSLog(@"[edgetam_coreml] unexpected output dtype %ld (want Float32)", (long)arr.dataType);
        return false;
    }
    __block bool ok = false;
    [arr getBytesWithHandler:^(const void* bytes, NSInteger size) {
        NSInteger want = count * (NSInteger)sizeof(float);
        if (size >= want) { memcpy(dst, bytes, want); ok = true; }
        else NSLog(@"[edgetam_coreml] output too small: %ld < %ld", (long)size, (long)want);
    }];
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
                          float* vision_features, float* hr0, float* hr1) {
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

        MLMultiArray* vf  = [[out featureValueForName:@"vision_features"] multiArrayValue];
        MLMultiArray* a0  = [[out featureValueForName:@"hr0"] multiArrayValue];
        MLMultiArray* a1  = [[out featureValueForName:@"hr1"] multiArrayValue];

        bool ok = copy_f32(vf, vision_features, 1 * 256 * 64 * 64)
               && copy_f32(a0, hr0,             1 * 32 * 256 * 256)
               && copy_f32(a1, hr1,             1 * 64 * 128 * 128);
        return ok ? 1 : 0;
    }
}

void edgetam_coreml_destroy(edgetam_coreml_handle handle) {
    auto* h = (EdgetamCoreML*)handle;
    if (h) { h->model = nil; delete h; }
}

}  // extern "C"
