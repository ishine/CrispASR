// crispasr_backend_gemma4_e2b.cpp — Gemma-4-E2B ASR backend adapter.

#include "crispasr_backend.h"
#include "gemma4_e2b.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class Gemma4E2BBackend : public CrispasrBackend {
public:
    Gemma4E2BBackend() = default;

    const char* name() const override { return "gemma4-e2b"; }

    uint32_t capabilities() const override { return CAP_LANGUAGE_DETECT; }

    bool init(const whisper_params& params) override {
        gemma4_e2b_context_params cp = gemma4_e2b_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        if (getenv("CRISPASR_VERBOSE") || getenv("GEMMA4_E2B_BENCH"))
            cp.verbosity = 2;
        cp.use_gpu = params.use_gpu;
        ctx_ = gemma4_e2b_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& /*params*/) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = gemma4_e2b_transcribe(ctx_, samples, n_samples);
        if (!text || !text[0]) {
            free(text);
            return out;
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text;
        free(text);

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            gemma4_e2b_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~Gemma4E2BBackend() override { Gemma4E2BBackend::shutdown(); }

private:
    gemma4_e2b_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_gemma4_e2b_backend() {
    return std::make_unique<Gemma4E2BBackend>();
}
