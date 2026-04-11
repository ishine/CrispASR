// crispasr_vad.cpp — implementation of Silero-VAD-backed audio slicing.
// See crispasr_vad.h for the interface contract.
//
// Extracted from examples/parakeet-main/main.cpp.

#include "crispasr_vad.h"
#include "whisper_params.h"

#include "whisper.h" // whisper_vad_* API

#include <algorithm>
#include <cstdio>

std::vector<crispasr_audio_slice> crispasr_compute_audio_slices(
    const float * samples,
    int           n_samples,
    int           sample_rate,
    int           chunk_seconds,
    const whisper_params & params)
{
    std::vector<crispasr_audio_slice> slices;

    whisper_vad_context * vctx = nullptr;
    const bool want_vad = !params.vad_model.empty();
    if (want_vad) {
        whisper_vad_context_params vcp = whisper_vad_default_context_params();
        vcp.n_threads = params.n_threads;
        vctx = whisper_vad_init_from_file_with_params(params.vad_model.c_str(), vcp);
        if (!vctx) {
            fprintf(stderr,
                    "crispasr: warning: failed to load VAD model '%s', "
                    "falling back to fixed chunking\n",
                    params.vad_model.c_str());
        }
    }

    if (vctx) {
        whisper_vad_params vp = whisper_vad_default_params();
        vp.threshold               = params.vad_threshold;
        vp.min_speech_duration_ms  = params.vad_min_speech_duration_ms;
        vp.min_silence_duration_ms = params.vad_min_silence_duration_ms;
        vp.speech_pad_ms           = (float) params.vad_speech_pad_ms;

        whisper_vad_segments * vseg =
            whisper_vad_segments_from_samples(vctx, vp, samples, n_samples);

        const int nv = vseg ? whisper_vad_segments_n_segments(vseg) : 0;
        for (int i = 0; i < nv; i++) {
            const float t0s = whisper_vad_segments_get_segment_t0(vseg, i);
            const float t1s = whisper_vad_segments_get_segment_t1(vseg, i);
            const int s = std::max(0, (int)(t0s * sample_rate));
            const int e = std::min(n_samples, (int)(t1s * sample_rate));
            if (e > s) {
                slices.push_back({
                    s, e,
                    (int64_t)(t0s * 100.0f),
                    (int64_t)(t1s * 100.0f),
                });
            }
        }
        if (vseg) whisper_vad_free_segments(vseg);
        whisper_vad_free(vctx);
        return slices;
    }

    // No VAD: fall back to fixed chunking. Encoders scale O(T^2) in frame
    // count, so unbounded audio hits memory/latency walls. The default
    // chunk_seconds (30 s) is a conservative window most models handle well.
    const int chunk_samples = chunk_seconds > 0 ? chunk_seconds * sample_rate
                                                : n_samples;

    if (n_samples <= chunk_samples) {
        const int64_t dur_cs = (int64_t)((double)n_samples / sample_rate * 100.0);
        slices.push_back({0, n_samples, 0, dur_cs});
    } else {
        for (int s = 0; s < n_samples; s += chunk_samples) {
            const int e = std::min(n_samples, s + chunk_samples);
            slices.push_back({
                s, e,
                (int64_t)((double)s / sample_rate * 100.0),
                (int64_t)((double)e / sample_rate * 100.0),
            });
        }
    }
    return slices;
}
