// RFD 0011 U8 — THREADED EdgeTAM tracker bench (encoder-ahead pipelining).
//
// The real tracker (preprocess + 3648-token assembly + 4 CoreML stages + bank) is
// ~69 ms/frame single-threaded (~14.5 fps). The encoder leg (preprocess + CoreML
// encode, ~17 ms) is bank-independent, so a producer thread runs it for frame N+1
// while the main thread runs the consumer (assembly + memattn + decode + memencode
// + bank, via sam3_propagate_frame) for frame N — overlapping the encoder to push
// toward 20 fps. The producer owns its own CoreML encoder handle (per-thread MLModel
// ownership); it hands each frame's 3 neck levels to the library via
// sam3_coreml_set_prefetched_neck, and the consumer's encode step consumes them
// (skipping preprocess + encode). Same goldeneval JSON as sam3_edgetam_bench.
//
// Run (all 4 stages on CoreML + PAD_BANK, like the single-threaded 14.5 fps config):
//   SAM3_COREML_PAD_BANK=1 SAM3_COREML_ENCODER=1 SAM3_COREML_MODEL=.../encoder.mlpackage \
//   SAM3_COREML_MEMATTN=1 SAM3_COREML_MEMATTN_MODEL=.../memattn.mlpackage \
//   SAM3_COREML_DECODER=1 SAM3_COREML_DECODER_MODEL=.../decoder.mlpackage \
//   SAM3_COREML_MEMENC=1 SAM3_COREML_MEMENC_MODEL=.../memencode.mlpackage \
//   sam3_coreml_tracker_threaded --models-dir <ggml> --video <mp4> --n-frames 120 \
//     --prompt <box.json> --dump-json <out.json>
#include "sam3.h"
#include "../coreml/edgetam_coreml.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

static std::string discover_edgetam_model(const std::string& dir) {
#ifndef _WIN32
    DIR* d = opendir(dir.c_str());
    if (!d) return "";
    std::string best;
    for (struct dirent* e; (e = readdir(d)) != nullptr;) {
        std::string f = e->d_name;
        if (f.size() > 5 && f.substr(f.size() - 5) == ".ggml" && f.find("edgetam") != std::string::npos) {
            best = dir + "/" + f; break;
        }
    }
    closedir(d);
    return best;
#else
    return dir + "/edgetam_f16.ggml";
#endif
}

static bool parse_box_json(const std::string& path, sam3_box& out) {
    std::ifstream f(path); if (!f) return false;
    std::stringstream ss; ss << f.rdbuf(); std::string s = ss.str();
    size_t lb = s.find('[', s.find("box")); if (lb == std::string::npos) return false;
    size_t rb = s.find(']', lb); if (rb == std::string::npos) return false;
    std::string in = s.substr(lb + 1, rb - lb - 1);
    for (char& c : in) if (c == ',') c = ' ';
    std::stringstream vs(in); float v[4]; int n = 0;
    while (n < 4 && (vs >> v[n])) ++n;
    if (n != 4) return false;
    out = {v[0], v[1], v[2], v[3]}; return true;
}

// Bounded blocking queue of pool-slot indices (SPSC producer<->consumer).
struct IQueue {
    std::queue<int> q; std::mutex m; std::condition_variable cv;
    void push(int v) { { std::lock_guard<std::mutex> l(m); q.push(v); } cv.notify_one(); }
    int pop() { std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return !q.empty(); }); int v = q.front(); q.pop(); return v; }
};

int main(int argc, char** argv) {
    std::string models_dir = "models/", video_path, prompt_path, dump_path;
    int n_frames = 120, warmup = 35;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--models-dir") models_dir = nx();
        else if (a == "--video") video_path = nx();
        else if (a == "--n-frames") n_frames = atoi(nx());
        else if (a == "--warmup") warmup = atoi(nx());
        else if (a == "--prompt") prompt_path = nx();
        else if (a == "--dump-json") dump_path = nx();
    }
    const char* enc_model = getenv("SAM3_COREML_MODEL");
    if (!enc_model) { fprintf(stderr, "ERROR: SAM3_COREML_MODEL (encoder .mlpackage) required\n"); return 1; }

    std::string model_path = discover_edgetam_model(models_dir);
    sam3_params params; params.model_path = model_path; params.use_gpu = true;
    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "ERROR: load model %s\n", model_path.c_str()); return 1; }
    auto state = sam3_create_state(*model, params);
    sam3_visual_track_params vtp; vtp.max_keep_alive = 1000; vtp.recondition_every = 16;
    auto tracker = sam3_create_visual_tracker(*model, vtp);

    auto vinfo = sam3_get_video_info(video_path);
    if (vinfo.n_frames <= 0) { fprintf(stderr, "ERROR: read video %s\n", video_path.c_str()); return 1; }
    if (n_frames > vinfo.n_frames) n_frames = vinfo.n_frames;
    sam3_box init_box;
    if (prompt_path.empty() || !parse_box_json(prompt_path, init_box)) {
        fprintf(stderr, "ERROR: --prompt box.json required\n"); return 1;
    }

    std::vector<sam3_image> frames(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        frames[f] = sam3_decode_video_frame(video_path, f);
        if (frames[f].data.empty()) { fprintf(stderr, "ERROR: decode frame %d\n", f); return 1; }
    }
    const float fw = (float)frames[0].width, fh = (float)frames[0].height;
    const int IMG = 1024;  // EdgeTAM encoder input size

    // Producer's own encoder handle (ANE) — per-thread MLModel ownership.
    auto prod_enc = edgetam_coreml_create(enc_model, /*ANE*/ 1);
    if (!prod_enc) { fprintf(stderr, "ERROR: producer encoder load %s\n", enc_model); return 1; }

    // Neck pool (channels-last [1,H,W,256]); POOL slots let the producer run ahead.
    const int POOL = 3, D = 256, Wd[3] = {256, 128, 64};
    struct Slot { std::vector<float> n[3]; int frame; };
    std::vector<Slot> pool(POOL);
    for (auto& s : pool) for (int i = 0; i < 3; ++i) s.n[i].assign((size_t)D * Wd[i] * Wd[i], 0.f);
    IQueue freeq, readyq;
    for (int i = 0; i < POOL; ++i) freeq.push(i);

    // Producer: preprocess + CoreML encode every frame (0..N-1), in order.
    std::thread producer([&] {
        for (int f = 0; f < n_frames; ++f) {
            int slot = freeq.pop();
            std::vector<float> img = sam3_coreml_preprocess_image(frames[f], IMG);
            edgetam_coreml_encode(prod_enc, img.data(),
                                  pool[slot].n[0].data(), pool[slot].n[1].data(), pool[slot].n[2].data());
            pool[slot].frame = f;
            readyq.push(slot);
        }
    });

    // Consumer (main): seed on frame 0, then propagate 1..N-1, all using prefetched necks.
    struct Rec { int frame; bool tracked; std::string state; float b[4]; float conf; };
    std::vector<Rec> recs; recs.reserve(n_frames);

    int slot0 = readyq.pop();  // frame 0
    sam3_coreml_set_prefetched_neck(pool[slot0].n[0].data(), pool[slot0].n[1].data(), pool[slot0].n[2].data());
    sam3_encode_image(*state, *model, frames[0]);
    (void)sam3_take_frame_timing();
    sam3_pvs_params pvs; pvs.box = init_box; pvs.use_box = true;
    int inst = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
    if (inst < 0) { fprintf(stderr, "ERROR: seed add_instance failed\n"); return 1; }
    (void)sam3_take_frame_timing();
    freeq.push(slot0);
    recs.push_back({0, true, "tracked", {init_box.x0 / fw, init_box.y0 / fh, init_box.x1 / fw, init_box.y1 / fh}, 1.f});

    auto wall0 = std::chrono::high_resolution_clock::now();
    for (int f = 1; f < n_frames; ++f) {
        int slot = readyq.pop();  // frame f (producer delivers in order)
        sam3_coreml_set_prefetched_neck(pool[slot].n[0].data(), pool[slot].n[1].data(), pool[slot].n[2].data());
        sam3_result res = sam3_propagate_frame(*tracker, *state, *model, frames[f]);
        (void)sam3_take_frame_timing();
        Rec r{f, false, "lost", {0, 0, 0, 0}, 0.f};
        if (!res.detections.empty()) {
            const sam3_detection* d = &res.detections[0];
            for (const auto& x : res.detections) if (x.instance_id == inst) { d = &x; break; }
            r.tracked = true; r.state = sam3_target_state_name(d->state);
            r.b[0] = d->box.x0 / fw; r.b[1] = d->box.y0 / fh; r.b[2] = d->box.x1 / fw; r.b[3] = d->box.y1 / fh;
            r.conf = d->score;
        }
        recs.push_back(r);
        freeq.push(slot);
    }
    auto wall1 = std::chrono::high_resolution_clock::now();
    producer.join();
    double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    double wall_fps = (n_frames - 1) * 1000.0 / wall_ms;

    printf("\nTHREADED tracker: %.2f fps  (%.1f ms over %d hold frames, encoder-ahead)\n",
           wall_fps, wall_ms, n_frames - 1);

    if (!dump_path.empty()) {
        std::ofstream o(dump_path);
        o << "{\"preds\":{";
        for (size_t i = 0; i < recs.size(); ++i) {
            const Rec& r = recs[i];
            o << (i ? "," : "") << "\"" << r.frame << "\":{\"bbox\":[" << r.b[0] << "," << r.b[1]
              << "," << r.b[2] << "," << r.b[3] << "],\"state\":\"" << r.state << "\"}";
        }
        o << "}}";
        fprintf(stderr, "wrote %s\n", dump_path.c_str());
    }
    edgetam_coreml_destroy(prod_enc);
    return 0;
}
