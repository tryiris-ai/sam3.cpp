cmake_minimum_required(VERSION 3.14)
project(sam3.cpp VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml options — enable Metal on Apple
option(SAM3_METAL "Enable Metal backend" ON)
if(APPLE AND SAM3_METAL)
    set(GGML_METAL        ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

# RFD 0011 (Windows/Linux cross-vendor GPU): ggml-Vulkan covers AMD/NVIDIA/Intel
# from one build. The Mac path uses Metal+CoreML; this is the non-Apple GPU path.
# The same cgo C-ABI (coreml/edgetam_capi.cpp) is reused below — its CoreML stage
# calls are #ifdef SAM3_COREML-guarded, so a Vulkan build runs the pure-ggml path
# (which lands on the Vulkan device). Requires the Vulkan SDK (glslc) at build.
option(SAM3_VULKAN "Enable the ggml-Vulkan cross-vendor GPU backend (non-Apple)" OFF)
if(NOT APPLE AND SAM3_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

# RFD 0011 (NVIDIA fast-path): ggml-CUDA runs the same EdgeTAM graph on NVIDIA GPUs
# with tensor-core FP16 kernels + CUDA graphs + FlashAttention. It is the NVIDIA
# sibling of the Vulkan (cross-vendor) and Metal/CoreML (Apple) backends. The cgo
# C-ABI (coreml/edgetam_capi.cpp) is reused below — its CoreML stage calls are
# #ifdef SAM3_COREML-guarded, so a CUDA build runs the pure-ggml path on the CUDA
# device. Setting GGML_CUDA=ON makes ggml define GGML_USE_CUDA PUBLIC (ggml_add_backend
# CUDA), which the model loader's #ifdef GGML_USE_CUDA branch keys on. Requires the
# CUDA Toolkit (nvcc) at build. SAM3_CUDA and SAM3_VULKAN may both be ON (fat build):
# the loader prefers CUDA then falls back to Vulkan then CPU.
option(SAM3_CUDA "Enable the ggml-CUDA backend (NVIDIA GPUs)" OFF)
if(NOT APPLE AND SAM3_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # CUDA graphs: standalone ggml defaults GGML_CUDA_GRAPHS to OFF (llama.cpp
    # flips it on in its own build). Without it, every frame pays per-kernel
    # launch overhead across the EdgeTAM graph's hundreds of small kernels.
    # Force it ON so the advertised "tensor-core FP16 + CUDA graphs" path is
    # actually compiled; ggml auto-falls back to normal kernel launches at
    # runtime for graphs it cannot capture, so this is safe.
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# sam3 static library
add_library(sam3 STATIC sam3.cpp sam3.h)
if(NOT APPLE AND SAM3_CUDA)
    # STEP-5 device frame preprocess (Windows parity plan v1): bit-exact
    # CUDA twin of sam3_resize_bilinear + ImageNet normalize. -fmad=false is
    # LOAD-BEARING for the byte gate (no FMA contraction in the double chain).
    enable_language(CUDA)
    target_sources(sam3 PRIVATE coreml/sam3_preproc.cu)
    set_source_files_properties(coreml/sam3_preproc.cu PROPERTIES LANGUAGE CUDA)
    set_property(SOURCE coreml/sam3_preproc.cu APPEND PROPERTY COMPILE_OPTIONS "-fmad=false")
    target_compile_definitions(sam3 PRIVATE SAM3_PREPROC_CUDA)
endif()
target_include_directories(sam3 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
)
target_link_libraries(sam3 PUBLIC ggml)
target_compile_features(sam3 PUBLIC cxx_std_14)

# RFD 0011 U0: per-frame "hold" timing instrumentation. OFF by default so the
# library has strictly zero timing overhead in normal builds (the SAM3_TIME_*
# macros expand to nothing and sam3_take_frame_timing() returns zeros). The
# RFD-0011 EdgeTAM bench requires this ON. PUBLIC so the definition propagates
# to consumers (e.g. sam3_edgetam_bench), keeping the bench's own copy of the
# HoldFrameTiming layout in agreement with the library's.
option(SAM3_TIMING "Compile per-frame hold-timing instrumentation into sam3" OFF)
if(SAM3_TIMING)
    target_compile_definitions(sam3 PUBLIC SAM3_TIMING)
endif()

# RFD 0011 (CoreML/ANE hybrid): run the EdgeTAM image encoder via CoreML on the
# Apple Neural Engine (~12 ms) instead of the ggml/Metal RepViT path (~122 ms).
# Apple-only; compiles the Objective-C++ bridge (coreml/edgetam_coreml.mm) and
# links the CoreML + Foundation frameworks. OFF by default. Even with this ON,
# the encoder path is gated at RUNTIME by the SAM3_COREML_ENCODER env var (+ the
# model path in SAM3_COREML_MODEL), so a SAM3_COREML build still defaults to ggml.
if(APPLE)
    option(SAM3_COREML "Build the CoreML/ANE EdgeTAM encoder bridge" OFF)
    if(SAM3_COREML)
        # Compile the Objective-C++ bridge as CXX with -x objective-c++ rather than
        # enable_language(OBJCXX). The latter (a) must sit after add_subdirectory(ggml)
        # or it reclassifies ggml's Metal .m files as C++ and breaks them, and (b) under
        # CMake 4.x, enabled in this conditional sub-scope, fails to emit a
        # CMAKE_OBJCXX_COMPILE_OBJECT rule ("cmake may not be built correctly"). Treating
        # the single .mm as CXX+objective-c++ sidesteps both and never touches ggml.
        target_sources(sam3 PRIVATE coreml/edgetam_coreml.mm)
        set_source_files_properties(coreml/edgetam_coreml.mm PROPERTIES
            LANGUAGE CXX
            COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
        # RFD 0011 U8: extern "C" tracker C-ABI for cgo into iris/platform.
        target_sources(sam3 PRIVATE coreml/edgetam_capi.cpp)
        target_include_directories(sam3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/coreml)
        target_compile_definitions(sam3 PUBLIC SAM3_COREML)
        target_link_libraries(sam3 PUBLIC "-framework CoreML" "-framework Foundation")
    endif()
endif()

# RFD 0011 (Windows/Vulkan + NVIDIA/CUDA): reuse the cgo tracker C-ABI on the
# non-Apple GPU paths WITHOUT the CoreML bridge. edgetam_capi.cpp's CoreML stage
# calls are #ifdef SAM3_COREML-guarded, so here the tracker runs the pure-ggml
# path on the Vulkan or CUDA device (with the opt-in ggml encoder-ahead producer
# on CUDA). SAM3_COREML is NOT defined, no .mm, no frameworks.
#
# SAM3_CAPI_SHARED: build the C-ABI as its own SHARED library (sam3capi.dll /
# libsam3capi.so) instead of folding it into the static libsam3. This is the
# Windows/CUDA cgo boundary: nvcc requires MSVC as its host compiler on Windows,
# and an MSVC-built C++ static lib cannot link into a mingw cgo binary (C++ ABI /
# runtime mismatch). A C-ABI DLL is toolchain-neutral — all C++ (sam3 + ggml +
# ggml-cuda + ggml-vulkan) stays inside the MSVC-built DLL and Go links only the
# 15 extern "C" entry points. Keep BUILD_SHARED_LIBS=OFF with this so ggml links
# statically INTO the DLL (one artifact + the CUDA runtime DLLs).
option(SAM3_CAPI_SHARED "Build the cgo C-ABI tracker as a shared library (Windows/MSVC cgo boundary)" OFF)
if(NOT APPLE AND (SAM3_VULKAN OR SAM3_CUDA))
    if(SAM3_CAPI_SHARED)
        add_library(sam3capi SHARED coreml/edgetam_capi.cpp)
        target_include_directories(sam3capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/coreml)
        # ET_CAPI_BUILD_DLL turns on __declspec(dllexport) for the capi entry
        # points (see ET_API in edgetam_capi.h) — extern "C" alone does not
        # export symbols from an MSVC DLL.
        target_compile_definitions(sam3capi PRIVATE ET_CAPI_BUILD_DLL)
        target_link_libraries(sam3capi PRIVATE sam3)
    else()
        target_sources(sam3 PRIVATE coreml/edgetam_capi.cpp)
        target_include_directories(sam3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/coreml)
    endif()
endif()

# Examples (only when top-level)
option(SAM3_BUILD_EXAMPLES "Build example executables" ON)
if(SAM3_BUILD_EXAMPLES AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(examples)
endif()

# Tests
option(SAM3_BUILD_TESTS "Build test executables" OFF)
if(SAM3_BUILD_TESTS)
    add_subdirectory(tests)
endif()
