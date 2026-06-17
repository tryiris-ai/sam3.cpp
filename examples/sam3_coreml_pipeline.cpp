// RFD 0011 — pure-CoreML 4-stage threaded pipeline (the real-time runtime core).
//
// Validates coreml/RUNTIME.md: a NO-GIL C++ encoder-ahead pipeline over the four
// CoreML stages should reach the consumer-bound throughput that Python threading
// capped at 23.7 fps (1.24x), because coremltools.predict() holds the GIL through
// its CPU-side marshalling while C++ does not.
//
// Structure (Egor's shape): the encoder (ANE) runs one frame ahead of the consumer
// (mem-attn GPU -> decoder -> memory-encode), connected by a bounded slot pool.
// Each model handle is owned by exactly one thread (encoder->producer; the rest->
// consumer), so no MLModel is touched concurrently. The memory bank is fixed/
// representative: throughput is shape-determined, and accuracy is validated
// separately by the hybrid + the parity-verified exports. This measures the
// THROUGHPUT thesis; the full host-side memory bank (SAMURAI selection, obj-ptr
// management) and cgo are the remaining RUNTIME.md pieces.
#include "../coreml/edgetam_coreml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <algorithm>

using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }

// minimal bounded blocking queue of slot indices (SPSC: producer<->consumer)
struct IQueue {
    std::queue<int> q; std::mutex m; std::condition_variable cv;
    void push(int v){ { std::lock_guard<std::mutex> l(m); q.push(v); } cv.notify_one(); }
    int  pop(){ std::unique_lock<std::mutex> l(m); cv.wait(l,[&]{return !q.empty();}); int v=q.front(); q.pop(); return v; }
};

static const char* envOr(const char* k, const char* d){ const char* v=getenv(k); return (v&&*v)?v:d; }
static int envUnit(const char* k, int d){ const char* v=getenv(k); return (v&&*v)?atoi(v):d; }  // 1=ANE 2=GPU 3=CPU

int main(int argc, char** argv){
    const int N      = argc>1 ? atoi(argv[1]) : 100;
    const int WARMUP = argc>2 ? atoi(argv[2]) : 15;
    const char* GE = envOr("SAM3_COREML_MODELS", "/Users/noah.johnson/iris/audits/goldenclip-eval/coreml_models");
    auto P=[&](const char* f){ std::string s=GE; s+="/"; s+=f; return s; };

    // heterogeneous placement (encoder ANE, mem-attn GPU; decoder/memenc default ANE, overridable)
    int U_DEC = envUnit("SAM3_DEC_UNIT_N", 1), U_ME = envUnit("SAM3_MENC_UNIT_N", 1);
    auto enc = edgetam_coreml_create(P("edgetam_encoder_neck_nhwc.mlpackage").c_str(), 1);
    auto ma  = edgetam_coreml_create(P("edgetam_memory_attention.mlpackage").c_str(),  2);
    auto dec = edgetam_coreml_create(P("edgetam_mask_decoder_nhwc.mlpackage").c_str(),  U_DEC);
    auto me  = edgetam_coreml_create(P("edgetam_memory_encode.mlpackage").c_str(),      U_ME);
    if(!enc||!ma||!dec||!me){ fprintf(stderr,"model load failed; set SAM3_COREML_MODELS=<dir with the 4 .mlpackages>\n"); return 1; }
    printf("loaded enc(ANE)+memattn(GPU)+dec(unit%d)+memenc(unit%d)  N=%d warmup=%d\n", U_DEC, U_ME, N, WARMUP);

    const size_t S_IMG=3*1024*1024, S_N0=256*256*256, S_N1=128*128*256, S_N2=64*64*256;
    const size_t S_CURR=4096*256, S_MEM=3648*64, S_PE=64*64*256, S_SP=256, S_MASKF=1024*1024;
    const size_t S_MASKS=4*256*256, S_TOK=4*256, S_MFEAT=512*64;

    std::vector<float> img(S_IMG,0.1f), memory(S_MEM,0.f), memory_pos(S_MEM,0.f), curr_pos(S_CURR,0.f);
    std::vector<float> image_pe(S_PE,0.f), sparse(S_SP,0.f), dense(S_PE,0.f), mask_full(S_MASKF,0.f);

    const int POOL=3;  // producer can be up to 2 frames ahead
    std::vector<std::vector<float>> n0(POOL), n1(POOL), n2(POOL);
    for(int i=0;i<POOL;i++){ n0[i].assign(S_N0,0); n1[i].assign(S_N1,0); n2[i].assign(S_N2,0); }
    std::vector<float> cond(S_CURR), masks(S_MASKS), iou(4), obj(1), tok(S_TOK), mfeat(S_MFEAT), mpos(S_MFEAT);

    auto consume=[&](int s){
        // curr == neck2 bytes ([4096,1,256] == [1,64,64,256]); mem-attn -> cond
        edgetam_coreml_memattn(ma, n2[s].data(), memory.data(), curr_pos.data(), memory_pos.data(), cond.data());
        edgetam_coreml_decode (dec, cond.data(), image_pe.data(), sparse.data(), dense.data(),
                               n0[s].data(), n1[s].data(), masks.data(), iou.data(), obj.data(), tok.data());
        // pix_feat fed from neck2 bytes (NHWC vs BCHW differ in value, not shape -> valid for timing)
        edgetam_coreml_memencode(me, n2[s].data(), mask_full.data(), mfeat.data(), mpos.data());
    };

    for(int i=0;i<WARMUP;i++){ edgetam_coreml_encode(enc,img.data(),n0[0].data(),n1[0].data(),n2[0].data()); consume(0); }

    // per-stage isolation timings (median of a short loop)
    auto stage_ms=[&](int reps, auto fn){ std::vector<double> v; for(int i=0;i<reps;i++){auto t=clk::now(); fn(); v.push_back(ms_since(t));} std::sort(v.begin(),v.end()); return v[v.size()/2]; };
    double t_enc = stage_ms(20, [&]{ edgetam_coreml_encode(enc,img.data(),n0[0].data(),n1[0].data(),n2[0].data()); });
    double t_ma  = stage_ms(20, [&]{ edgetam_coreml_memattn(ma,n2[0].data(),memory.data(),curr_pos.data(),memory_pos.data(),cond.data()); });
    double t_dec = stage_ms(20, [&]{ edgetam_coreml_decode(dec,cond.data(),image_pe.data(),sparse.data(),dense.data(),n0[0].data(),n1[0].data(),masks.data(),iou.data(),obj.data(),tok.data()); });
    double t_me  = stage_ms(20, [&]{ edgetam_coreml_memencode(me,n2[0].data(),mask_full.data(),mfeat.data(),mpos.data()); });
    printf("per-stage: enc %.1f  memattn %.1f  dec %.1f  memenc %.1f ms  (consumer=%.1f ms)\n",
           t_enc,t_ma,t_dec,t_me, t_ma+t_dec+t_me);

    // sequential baseline
    auto t0=clk::now();
    for(int i=0;i<N;i++){ edgetam_coreml_encode(enc,img.data(),n0[0].data(),n1[0].data(),n2[0].data()); consume(0); }
    double seq_fps = 1000.0*N/ms_since(t0);

    // threaded encoder-ahead pipeline
    IQueue freeq, readyq;
    for(int i=0;i<POOL;i++) freeq.push(i);
    auto tt0=clk::now();
    std::thread producer([&]{ for(int k=0;k<N;k++){ int s=freeq.pop(); edgetam_coreml_encode(enc,img.data(),n0[s].data(),n1[s].data(),n2[s].data()); readyq.push(s);} readyq.push(-1); });
    std::thread consumer([&]{ for(;;){ int s=readyq.pop(); if(s<0) break; consume(s); freeq.push(s); } });
    producer.join(); consumer.join();
    double thr_fps = 1000.0*N/ms_since(tt0);

    printf("\nsequential 4-stage: %.1f fps\n", seq_fps);
    printf("threaded  4-stage: %.1f fps   (speedup %.2fx vs sequential; Python threaded was 1.24x, GIL-bound)\n",
           thr_fps, thr_fps/seq_fps);

    edgetam_coreml_destroy(enc); edgetam_coreml_destroy(ma); edgetam_coreml_destroy(dec); edgetam_coreml_destroy(me);
    return 0;
}
