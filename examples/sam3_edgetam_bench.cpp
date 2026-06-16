/**
 * sam3_edgetam_bench — per-frame video-replay benchmark + JSON dump for the
 * EdgeTAM video tracker (RFD 0011 unit U0).
 *
 * Unlike sam3_profile_edgetam (single-image, sub-graph breakdown) this drives
 * the FULL tracker frame-by-frame on a video — the steady-state "hold" loop —
 * and records, for every frame, the fine-grained HoldFrameTiming breakdown
 * (per-stage build / alloc / compute, split apart) plus the predicted bbox.
 *
 * WHY THE BUILD/ALLOC/COMPUTE SPLIT: RFD 0011 U1 removes the per-frame ggml
 * graph rebuild; its acceptance test is "graph build_ms + alloc_ms ≈ 0 in
 * steady state". That is only checkable because this bench (and the library
 * instrumentation it reads via sam3_take_frame_timing) keeps build, alloc, and
 * compute as DISTINCT numbers per stage. This bench produces the U0 baseline
 * that U1 asserts against.
 *
 * EdgeTAM is a visual-only model, so the tracker is created with
 * sam3_create_visual_tracker and frames are advanced with sam3_propagate_frame
 * (sam3_track_frame is unavailable on visual-only models). This mirrors the
 * visual-only branch of examples/benchmark.cpp.
 *
 * Usage:
 *   sam3_edgetam_bench [options]
 *
 * Options:
 *   --models-dir <path>   Directory with .ggml files   (default: models/)
 *   --video <path>        Video file                    (default: data/test_video.mp4)
 *   --n-frames <n>        Frames to process incl. seed  (default: 30)
 *   --warmup <n>          Warmup propagated frames excluded from stats (default: 5)
 *   --n-threads <n>       CPU threads                   (default: 4)
 *   --cpu-only            Force CPU backend
 *   --gpu-only            Force Metal/GPU backend (default behavior)
 *   --prompt <box.json>   Seed box JSON {"box":[x0,y0,x1,y1]} in PIXELS
 *                         (if absent, a centered default box is used)
 *   --dump-json <path>    Write per-frame predictions + timing to <path>
 *   --resolution <n>      Override encoder input resolution (0 = model default)
 */

#include "sam3.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <sys/stat.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

// ── Helpers ──────────────────────────────────────────────────────────────────

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Discover the first EdgeTAM .ggml model in a directory (filter "edgetam").
// EdgeTAM is the only model this bench measures, so we pick the first match.
static std::string discover_edgetam_model(const std::string& dir) {
#ifdef _WIN32
    std::string pattern = dir + "\\*.ggml";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
    if (hFind == INVALID_HANDLE_VALUE) return "";
    std::string best;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        std::string fname = fd.cFileName;
        if (!ends_with(fname, ".ggml")) continue;
        if (fname.find("edgetam") == std::string::npos) continue;
        best = dir + "\\" + fname;
        break;
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
    return best;
#else
    DIR* d = opendir(dir.c_str());
    if (!d) return "";
    std::string best;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string fname = ent->d_name;
        if (!ends_with(fname, ".ggml")) continue;
        if (fname.find("edgetam") == std::string::npos) continue;
        best = dir + "/" + fname;
        break;
    }
    closedir(d);
    return best;
#endif
}

// Minimal extractor: pull the first JSON array of 4 numbers found after the
// "box" key in a tiny prompt file like {"box":[x0,y0,x1,y1]}. We avoid pulling
// in a JSON dependency for a 4-number seed; the format is fixed and simple.
static bool parse_box_json(const std::string& path, sam3_box& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    size_t key = s.find("box");
    size_t lb = s.find('[', key == std::string::npos ? 0 : key);
    if (lb == std::string::npos) return false;
    size_t rb = s.find(']', lb);
    if (rb == std::string::npos) return false;
    std::string inner = s.substr(lb + 1, rb - lb - 1);
    for (char& c : inner) if (c == ',') c = ' ';
    std::stringstream vs(inner);
    float v[4];
    int n = 0;
    while (n < 4 && (vs >> v[n])) ++n;
    if (n != 4) return false;
    out = {v[0], v[1], v[2], v[3]};
    return true;
}

// Per-frame record collected during the run.
struct FrameRecord {
    int    frame_id    = 0;
    bool   tracked     = false;
    std::string state  = "lost";         // RFD 0011 U4: lifecycle state name
    float  bbox[4]     = {0, 0, 0, 0};  // corner-normalized [x0,y0,x1,y1] in [0,1]
    float  confidence  = 0.0f;
    float  object_score = 0.0f;
    float  mask_iou_pred = 0.0f;
    bool   is_seed     = false;          // frame 0 (no propagation timing)
    HoldFrameTiming timing;
};

// Accumulator for one HoldFrameTiming field across post-warmup frames.
struct StageStats {
    std::vector<double> samples;
    void add(double v) { samples.push_back(v); }
    double mean() const {
        if (samples.empty()) return 0.0;
        double s = 0.0;
        for (double v : samples) s += v;
        return s / samples.size();
    }
    double p95() const {
        if (samples.empty()) return 0.0;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        // Nearest-rank p95 (index ceil(0.95*N)-1), clamped.
        size_t idx = (size_t)std::ceil(0.95 * sorted.size());
        if (idx == 0) idx = 1;
        if (idx > sorted.size()) idx = sorted.size();
        return sorted[idx - 1];
    }
};

// Escape nothing fancy — our strings are plain ASCII keys/paths. Numbers are
// emitted with enough precision to round-trip the IoU scorer.
static void emit_timing_json(std::ostream& os, const HoldFrameTiming& t) {
    char buf[1400];
    snprintf(buf, sizeof(buf),
        "{\"preprocess_ms\":%.4f,"
        "\"image_encoder_build_ms\":%.4f,\"image_encoder_alloc_ms\":%.4f,\"image_encoder_compute_ms\":%.4f,"
        "\"mem_attn_build_ms\":%.4f,\"mem_attn_alloc_ms\":%.4f,\"mem_attn_compute_ms\":%.4f,"
        "\"mask_decoder_compute_ms\":%.4f,"
        "\"mem_encoder_build_ms\":%.4f,\"mem_encoder_alloc_ms\":%.4f,\"mem_encoder_compute_ms\":%.4f,"
        "\"memory_bank_update_ms\":%.4f,\"mask_to_bbox_ms\":%.4f,\"state_update_ms\":%.4f,"
        "\"total_ms\":%.4f}",
        t.preprocess_ms,
        t.image_encoder_build_ms, t.image_encoder_alloc_ms, t.image_encoder_compute_ms,
        t.mem_attn_build_ms, t.mem_attn_alloc_ms, t.mem_attn_compute_ms,
        t.mask_decoder_compute_ms,
        t.mem_encoder_build_ms, t.mem_encoder_alloc_ms, t.mem_encoder_compute_ms,
        t.memory_bank_update_ms, t.mask_to_bbox_ms, t.state_update_ms,
        t.total_ms);
    os << buf;
}

// ── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string models_dir = "models/";
    std::string video_path = "data/test_video.mp4";
    std::string prompt_path;
    std::string dump_json_path;
    int  n_frames   = 30;
    int  warmup     = 5;
    int  n_threads  = 4;
    int  resolution = 0;     // 0 = model default
    bool cpu_only   = false;
    bool gpu_only   = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--models-dir" && i + 1 < argc) { models_dir = argv[++i]; }
        else if (arg == "--video"      && i + 1 < argc) { video_path = argv[++i]; }
        else if (arg == "--n-frames"   && i + 1 < argc) { n_frames   = atoi(argv[++i]); }
        else if (arg == "--warmup"     && i + 1 < argc) { warmup     = atoi(argv[++i]); }
        else if (arg == "--n-threads"  && i + 1 < argc) { n_threads  = atoi(argv[++i]); }
        else if (arg == "--prompt"     && i + 1 < argc) { prompt_path = argv[++i]; }
        else if (arg == "--dump-json"  && i + 1 < argc) { dump_json_path = argv[++i]; }
        else if (arg == "--resolution" && i + 1 < argc) { resolution = atoi(argv[++i]); }
        else if (arg == "--cpu-only")  { cpu_only = true; }
        else if (arg == "--gpu-only")  { gpu_only = true; }
        else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: %s [options]\n"
                "  --models-dir <path>   Models directory          (default: models/)\n"
                "  --video <path>        Video file                 (default: data/test_video.mp4)\n"
                "  --n-frames <n>        Frames incl. seed          (default: 30)\n"
                "  --warmup <n>          Warmup propagated frames   (default: 5)\n"
                "  --n-threads <n>       CPU threads                (default: 4)\n"
                "  --cpu-only            Force CPU backend\n"
                "  --gpu-only            Force Metal/GPU backend\n"
                "  --prompt <box.json>   Seed box {\"box\":[x0,y0,x1,y1]} in PIXELS\n"
                "  --dump-json <path>    Per-frame predictions + timing JSON\n"
                "  --resolution <n>      Encoder input resolution   (0 = model default)\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (cpu_only && gpu_only) {
        fprintf(stderr, "ERROR: --cpu-only and --gpu-only are mutually exclusive\n");
        return 1;
    }
    if (n_frames < 2) {
        fprintf(stderr, "ERROR: --n-frames must be >= 2 (one seed + at least one hold frame)\n");
        return 1;
    }
    // Default backend is GPU/Metal; --cpu-only switches to CPU.
    bool use_gpu = !cpu_only;

    // ── Locate EdgeTAM model ─────────────────────────────────────────────
    std::string model_path = discover_edgetam_model(models_dir);
    if (model_path.empty()) {
        fprintf(stderr, "ERROR: no EdgeTAM (*edgetam*.ggml) model found in '%s'\n",
                models_dir.c_str());
        return 1;
    }
    fprintf(stderr, "Model:    %s\n", model_path.c_str());
    fprintf(stderr, "Video:    %s\n", video_path.c_str());
    fprintf(stderr, "Backend:  %s\n", use_gpu ? "Metal (GPU)" : "CPU");

    // ── Video info + frame clamp ─────────────────────────────────────────
    auto vinfo = sam3_get_video_info(video_path);
    if (vinfo.n_frames <= 0) {
        fprintf(stderr, "ERROR: cannot read video '%s'\n", video_path.c_str());
        return 1;
    }
    if (n_frames > vinfo.n_frames) {
        fprintf(stderr, "WARNING: video has %d frames, clamping --n-frames to %d\n",
                vinfo.n_frames, vinfo.n_frames);
        n_frames = vinfo.n_frames;
    }
    fprintf(stderr, "Video:    %dx%d, %d frames, %.1f fps\n",
            vinfo.width, vinfo.height, vinfo.n_frames, vinfo.fps);
    if (warmup >= n_frames - 1) {
        fprintf(stderr, "WARNING: --warmup %d >= hold frames (%d); clamping to %d\n",
                warmup, n_frames - 1, n_frames - 2);
        warmup = n_frames - 2;
        if (warmup < 0) warmup = 0;
    }

    // ── Seed box (pixels, original-frame coords) ─────────────────────────
    sam3_box init_box;
    if (!prompt_path.empty()) {
        if (!parse_box_json(prompt_path, init_box)) {
            fprintf(stderr, "ERROR: failed to parse seed box from '%s' "
                    "(expected {\"box\":[x0,y0,x1,y1]})\n", prompt_path.c_str());
            return 1;
        }
    } else {
        // Centered default box covering the middle ~40% of the frame.
        float cw = vinfo.width * 0.2f, ch = vinfo.height * 0.2f;
        float cx = vinfo.width * 0.5f, cy = vinfo.height * 0.5f;
        init_box = {cx - cw, cy - ch, cx + cw, cy + ch};
    }
    fprintf(stderr, "Seed box: [%.1f, %.1f, %.1f, %.1f] (pixels)\n",
            init_box.x0, init_box.y0, init_box.x1, init_box.y1);

    // ── Load model + state + visual-only tracker ─────────────────────────
    sam3_params params;
    params.model_path     = model_path;
    params.use_gpu        = use_gpu;
    params.n_threads      = n_threads;
    params.encode_img_size = resolution;  // 0 = model default

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }
    if (sam3_get_model_type(*model) != SAM3_MODEL_EDGETAM) {
        fprintf(stderr, "ERROR: model is not EdgeTAM (model_type=%d)\n",
                sam3_get_model_type(*model));
        return 1;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "ERROR: failed to create state\n");
        return 1;
    }

    // EdgeTAM is visual-only: create a visual tracker and seed the instance
    // manually, then advance with sam3_propagate_frame (per benchmark.cpp).
    sam3_visual_track_params vtp;
    vtp.max_keep_alive    = 1000;   // keep the instance alive for the whole clip
    vtp.recondition_every = 16;
    auto tracker = sam3_create_visual_tracker(*model, vtp);
    if (!tracker) {
        fprintf(stderr, "ERROR: failed to create visual tracker\n");
        return 1;
    }

    // ── Decode all frames up front (frame-accurate; matches benchmark.cpp) ─
    std::vector<sam3_image> frames(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        frames[f] = sam3_decode_video_frame(video_path, f);
        if (frames[f].data.empty()) {
            fprintf(stderr, "ERROR: failed to decode frame %d\n", f);
            return 1;
        }
    }
    const float fw = (float)frames[0].width;
    const float fh = (float)frames[0].height;

    std::vector<FrameRecord> records;
    records.reserve(n_frames);

    // ── Frame 0: encode + seed instance from the box prompt ──────────────
    if (!sam3_encode_image(*state, *model, frames[0])) {
        fprintf(stderr, "ERROR: encode of seed frame failed\n");
        return 1;
    }
    // Drain any timing accumulated by the seed-frame encode so it does not
    // bleed into frame 1's record.
    (void)sam3_take_frame_timing();

    sam3_pvs_params pvs;
    pvs.box       = init_box;
    pvs.use_box   = true;
    pvs.multimask = false;
    int inst_id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
    if (inst_id < 0) {
        fprintf(stderr, "ERROR: add_instance from seed box failed\n");
        return 1;
    }
    (void)sam3_take_frame_timing();  // discard seed memory-encode timing

    {
        // Frame 0's "prediction" is the seed prompt box itself (normalized).
        FrameRecord r0;
        r0.frame_id     = 0;
        r0.tracked      = true;
        r0.state        = "tracked";  // RFD 0011 U4: seed frame is tracked by definition
        r0.is_seed      = true;
        r0.bbox[0]      = init_box.x0 / fw;
        r0.bbox[1]      = init_box.y0 / fh;
        r0.bbox[2]      = init_box.x1 / fw;
        r0.bbox[3]      = init_box.y1 / fh;
        r0.confidence   = 1.0f;
        records.push_back(r0);
    }

    // ── Frames 1..N-1: hold loop (propagate + per-frame timing) ──────────
    auto wall_start = std::chrono::high_resolution_clock::now();
    for (int f = 1; f < n_frames; ++f) {
        sam3_result res = sam3_propagate_frame(*tracker, *state, *model, frames[f]);
        HoldFrameTiming t = sam3_take_frame_timing();

        FrameRecord r;
        r.frame_id = f;
        r.timing   = t;
        if (!res.detections.empty()) {
            // Pick the detection for our seeded instance (or the first one).
            const sam3_detection* det = &res.detections[0];
            for (const auto& d : res.detections) {
                if (d.instance_id == inst_id) { det = &d; break; }
            }
            r.tracked       = true;
            r.state         = sam3_target_state_name(det->state);  // RFD 0011 U4
            // det.box is in original-frame PIXEL corners; normalize to [0,1].
            r.bbox[0]       = det->box.x0 / fw;
            r.bbox[1]       = det->box.y0 / fh;
            r.bbox[2]       = det->box.x1 / fw;
            r.bbox[3]       = det->box.y1 / fh;
            r.confidence    = det->score;
            r.object_score  = det->mask.obj_score;
            r.mask_iou_pred = det->mask.iou_score;
        }
        records.push_back(r);

        fprintf(stderr, "  frame %2d/%d  total=%7.1f ms  imgEnc(c)=%6.1f  memAttn(c)=%6.1f  memEnc(c)=%6.1f  %s\n",
                f, n_frames - 1, t.total_ms,
                t.image_encoder_compute_ms, t.mem_attn_compute_ms, t.mem_encoder_compute_ms,
                r.tracked ? "tracked" : "LOST");
    }
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // ── Aggregate per-stage stats over post-warmup hold frames ───────────
    // Hold frames are records[1 .. n_frames-1]; skip the first `warmup` of them.
    std::map<std::string, StageStats> stats;
    auto accumulate = [&](const HoldFrameTiming& t) {
        stats["preprocess"].add(t.preprocess_ms);
        stats["image_encoder_build"].add(t.image_encoder_build_ms);
        stats["image_encoder_alloc"].add(t.image_encoder_alloc_ms);
        stats["image_encoder_compute"].add(t.image_encoder_compute_ms);
        stats["mem_attn_build"].add(t.mem_attn_build_ms);
        stats["mem_attn_alloc"].add(t.mem_attn_alloc_ms);
        stats["mem_attn_compute"].add(t.mem_attn_compute_ms);
        stats["mask_decoder_compute"].add(t.mask_decoder_compute_ms);
        stats["mem_encoder_build"].add(t.mem_encoder_build_ms);
        stats["mem_encoder_alloc"].add(t.mem_encoder_alloc_ms);
        stats["mem_encoder_compute"].add(t.mem_encoder_compute_ms);
        stats["memory_bank_update"].add(t.memory_bank_update_ms);
        stats["mask_to_bbox"].add(t.mask_to_bbox_ms);
        stats["state_update"].add(t.state_update_ms);
        stats["total"].add(t.total_ms);
    };

    int counted = 0;
    for (size_t i = 1; i < records.size(); ++i) {
        if (records[i].is_seed) continue;
        int hold_index = (int)i - 1;          // 0-based index among hold frames
        if (hold_index < warmup) continue;    // skip warmup
        accumulate(records[i].timing);
        ++counted;
    }

    // FPS from post-warmup hold frames using their measured total_ms.
    double mean_total = stats["total"].mean();
    double fps = (mean_total > 0.0) ? 1000.0 / mean_total : 0.0;
    // Wall FPS over ALL hold frames (incl. warmup + ffmpeg-free compute only).
    double wall_fps = (wall_ms > 0.0) ? (double)(n_frames - 1) * 1000.0 / wall_ms : 0.0;

    // ── Human summary table ──────────────────────────────────────────────
    printf("\n");
    printf("=================================================================================\n");
    printf("SAM3 EdgeTAM bench — %s, %d frames (1 seed + %d hold), warmup=%d, counted=%d\n",
           use_gpu ? "Metal" : "CPU", n_frames, n_frames - 1, warmup, counted);
    printf("model: %s\n", model_path.c_str());
    printf("=================================================================================\n\n");
    printf("  %-24s | %12s | %12s\n", "Stage", "mean (ms)", "p95 (ms)");
    printf("  -------------------------+--------------+-------------\n");
    const char* order[] = {
        "preprocess",
        "image_encoder_build", "image_encoder_alloc", "image_encoder_compute",
        "mem_attn_build", "mem_attn_alloc", "mem_attn_compute",
        "mask_decoder_compute",
        "mem_encoder_build", "mem_encoder_alloc", "mem_encoder_compute",
        "memory_bank_update", "mask_to_bbox", "state_update",
        "total",
    };
    for (const char* name : order) {
        printf("  %-24s | %12.3f | %12.3f\n",
               name, stats[name].mean(), stats[name].p95());
    }
    printf("  -------------------------+--------------+-------------\n");
    printf("\n  total ms/frame (mean): %.3f\n", mean_total);
    printf("  fps (from mean total): %.2f\n", fps);
    printf("  wall fps (hold loop):  %.2f  (%.1f ms over %d hold frames)\n",
           wall_fps, wall_ms, n_frames - 1);

    // ── JSON dump ────────────────────────────────────────────────────────
    if (!dump_json_path.empty()) {
        std::ofstream js(dump_json_path);
        if (!js) {
            fprintf(stderr, "ERROR: cannot open --dump-json path '%s'\n",
                    dump_json_path.c_str());
            return 1;
        }
        // Sequence name: video basename without extension.
        std::string seq = video_path;
        size_t slash = seq.find_last_of("/\\");
        if (slash != std::string::npos) seq = seq.substr(slash + 1);
        size_t dot = seq.find_last_of('.');
        if (dot != std::string::npos) seq = seq.substr(0, dot);

        js << "{\"seq\":\"" << seq << "\",";
        js << "\"nfr\":" << n_frames << ",";
        js << "\"fps\":" << fps << ",";
        js << "\"preds\":{";
        for (size_t i = 0; i < records.size(); ++i) {
            const FrameRecord& r = records[i];
            if (i) js << ",";
            js << "\"" << r.frame_id << "\":{";
            js << "\"frame_id\":" << r.frame_id << ",";
            js << "\"state\":\"" << r.state << "\",";  // RFD 0011 U4: real lifecycle state
            char bbuf[128];
            snprintf(bbuf, sizeof(bbuf), "\"bbox\":[%.6f,%.6f,%.6f,%.6f],",
                     r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]);
            js << bbuf;
            char sbuf[160];
            snprintf(sbuf, sizeof(sbuf),
                     "\"confidence\":%.6f,\"object_score\":%.6f,\"mask_iou_pred\":%.6f,",
                     r.confidence, r.object_score, r.mask_iou_pred);
            js << sbuf;
            js << "\"timing_ms\":";
            emit_timing_json(js, r.timing);
            js << "}";
        }
        js << "}}";
        js.close();
        fprintf(stderr, "\nWrote per-frame JSON: %s\n", dump_json_path.c_str());
    }

    return 0;
}
