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

add_subdirectory(ggml)

# sam3 static library
add_library(sam3 STATIC sam3.cpp sam3.h)
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
