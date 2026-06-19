// RFD 0011 U8 Wave 5 — verify the capi encoder-ahead split end-to-end through the
// SAME C-ABI cgo will call (edgetam_capi_create/seed/encode_slot/track_slot), so a
// green fps + goldeneval here de-risks the Go wiring. Producer thread: encode_slot;
// consumer (main): track_slot. Same {"preds":{...}} JSON as sam3_edgetam_bench.
//
// Run (CoreML dir holds the 4 .mlpackages; ggml model is the seed checkpoint):
//   sam3_coreml_capi_threaded --models-dir-coreml <dir> --ggml-model <edgetam_f16.ggml> \
//     --video <mp4> --prompt <box.json> --n-frames 120 --dump-json <out.json>
#include "../coreml/edgetam_capi.h"
#include "sam3.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static bool parse_box_json(const std::string& path, et_box& out) {
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

static const char* state_name(int s) {
    switch (s) { case 0: return "tracked"; case 1: return "at_risk"; case 2: return "occluded";
                 case 3: return "lost";    case 4: return "candidate_reacquire"; case 5: return "reacquired";
                 default: return "lost"; }
}

// Bounded blocking queue of pool-slot ids (SPSC producer<->consumer).
struct IQueue {
    std::queue<int> q; std::mutex m; std::condition_variable cv;
    void push(int v) { { std::lock_guard<std::mutex> l(m); q.push(v); } cv.notify_one(); }
    int pop() { std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return !q.empty(); }); int v = q.front(); q.pop(); return v; }
};

int main(int argc, char** argv) {
    std::string coreml_dir, ggml_model = "models/edgetam_f16.ggml", video, prompt, dump;
    int n_frames = 120;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--models-dir-coreml") coreml_dir = nx();
        else if (a == "--ggml-model") ggml_model = nx();
        else if (a == "--video") video = nx();
        else if (a == "--prompt") prompt = nx();
        else if (a == "--n-frames") n_frames = atoi(nx());
        else if (a == "--dump-json") dump = nx();
    }
    if (coreml_dir.empty() || video.empty() || prompt.empty()) {
        fprintf(stderr, "need --models-dir-coreml --video --prompt\n"); return 1;
    }

    edgetam_tracker_t h = edgetam_capi_create(coreml_dir.c_str(), ggml_model.c_str(), 1, 1);
    if (!h) { fprintf(stderr, "ERROR: edgetam_capi_create failed\n"); return 1; }
    const int POOL = edgetam_capi_pool_size(h);
    if (POOL <= 0) { fprintf(stderr, "ERROR: threading unavailable (CoreML off?)\n"); return 1; }

    auto vinfo = sam3_get_video_info(video);
    if (vinfo.n_frames <= 0) { fprintf(stderr, "ERROR: read video\n"); return 1; }
    if (n_frames > vinfo.n_frames) n_frames = vinfo.n_frames;
    et_box init_box;
    if (!parse_box_json(prompt, init_box)) { fprintf(stderr, "ERROR: --prompt box.json\n"); return 1; }

    std::vector<sam3_image> frames(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        frames[f] = sam3_decode_video_frame(video, f);
        if (frames[f].data.empty()) { fprintf(stderr, "ERROR: decode frame %d\n", f); return 1; }
    }
    const int w = frames[0].width, hgt = frames[0].height;
    const float fw = (float)w, fh = (float)hgt;

    if (!edgetam_capi_seed(h, frames[0].data.data(), w, hgt, init_box)) {
        fprintf(stderr, "ERROR: seed failed\n"); return 1;
    }

    struct Rec { int frame; std::string state; float b[4]; };
    std::vector<Rec> recs; recs.reserve(n_frames);
    recs.push_back({0, "tracked", {init_box.x0 / fw, init_box.y0 / fh, init_box.x1 / fw, init_box.y1 / fh}});

    IQueue freeq, readyq;
    for (int i = 0; i < POOL; ++i) freeq.push(i);

    // Producer: encode frames 1..N-1 in order into free slots.
    std::thread producer([&] {
        for (int f = 1; f < n_frames; ++f) {
            int slot = freeq.pop();
            edgetam_capi_encode_slot(h, slot, frames[f].data.data(), w, hgt);
            readyq.push(slot);
        }
    });

    // Consumer (main): track each frame using its prefetched neck.
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int f = 1; f < n_frames; ++f) {
        int slot = readyq.pop();
        et_result r = edgetam_capi_track_slot(h, slot, frames[f].data.data(), w, hgt);
        Rec rec{f, r.valid ? state_name(r.state) : "lost",
                {r.box.x0, r.box.y0, r.box.x1, r.box.y1}};
        recs.push_back(rec);
        freeq.push(slot);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    producer.join();

    // RFD 0011 U8 mask export check: pull the last track's mask via the C-ABI.
    {
        int mw = 0, mh = 0;
        int need = edgetam_capi_last_mask(h, nullptr, 0, &mw, &mh);
        if (need > 0) {
            std::vector<uint8_t> mbuf(need);
            edgetam_capi_last_mask(h, mbuf.data(), need, &mw, &mh);
            size_t fg = 0; for (auto v : mbuf) if (v) ++fg;
            printf("MASK export: last frame %dx%d, %zu foreground px (%.1f%% of frame)\n",
                   mw, mh, fg, 100.0 * (double)fg / (double)need);
        } else {
            printf("MASK export: EMPTY on last frame (w=%d h=%d) — mask not populated on cgo path\n", mw, mh);
        }
    }
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("\nCAPI THREADED: %.2f fps  (%.1f ms over %d hold frames, encoder-ahead via capi)\n",
           (n_frames - 1) * 1000.0 / wall_ms, wall_ms, n_frames - 1);

    if (!dump.empty()) {
        std::ofstream o(dump);
        o << "{\"preds\":{";
        for (size_t i = 0; i < recs.size(); ++i) {
            const Rec& r = recs[i];
            o << (i ? "," : "") << "\"" << r.frame << "\":{\"bbox\":[" << r.b[0] << "," << r.b[1]
              << "," << r.b[2] << "," << r.b[3] << "],\"state\":\"" << r.state << "\"}";
        }
        o << "}}";
        fprintf(stderr, "wrote %s\n", dump.c_str());
    }
    edgetam_capi_destroy(h);
    return 0;
}
