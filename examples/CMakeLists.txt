# Example executables for sam3.cpp

# Quantization tool (no GUI dependency)
add_executable(sam3_quantize quantize.cpp)
target_link_libraries(sam3_quantize PRIVATE ggml)

add_executable(sam3_benchmark benchmark.cpp)
target_link_libraries(sam3_benchmark PRIVATE sam3)

add_executable(sam3_profile_edgetam profile_edgetam.cpp)
target_link_libraries(sam3_profile_edgetam PRIVATE sam3)

# RFD 0011 U0: per-frame EdgeTAM video-replay bench + JSON dump. Requires the
# HoldFrameTiming instrumentation, so it compiles with SAM3_TIMING. The library
# must ALSO be built with -DSAM3_TIMING (top-level option SAM3_TIMING=ON) for
# the timers inside sam3.cpp to be active; this PRIVATE definition keeps the
# bench's own translation unit consistent and self-documents the requirement.
add_executable(sam3_edgetam_bench sam3_edgetam_bench.cpp)
target_link_libraries(sam3_edgetam_bench PRIVATE sam3)
target_compile_definitions(sam3_edgetam_bench PRIVATE SAM3_TIMING)

# SDL2 + ImGui are optional — build only if SDL2 is found.
find_package(SDL2 QUIET)

if(SDL2_FOUND)
    add_subdirectory(third-party)

    add_executable(sam3_image main_image.cpp)
    target_link_libraries(sam3_image PRIVATE sam3 imgui-sdl2 SDL2::SDL2)

    add_executable(sam3_video main_video.cpp)
    target_link_libraries(sam3_video PRIVATE sam3 imgui-sdl2 SDL2::SDL2)
else()
    message(STATUS "SDL2 not found — skipping GUI examples")
endif()
