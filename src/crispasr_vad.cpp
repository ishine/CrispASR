// crispasr_vad.cpp — Silero VAD segmentation + stitching for shared use.
// See crispasr_vad.h for the interface contract.
//
// Extracted from examples/cli/crispasr_vad.cpp so that the CLI, the C-ABI
// wrapper `crispasr_session_transcribe_vad`, and every language binding
// use the same implementation. Auto-download / cache resolution stays in
// the CLI (it's a UX policy, not a library responsibility) — this file
// operates on a concrete VAD model path supplied by the caller.

#include "crispasr_vad.h"

#include "whisper.h" // whisper_vad_* API

#include <algorithm>
#include <cstdio>
#include <cstring>

std::vector<crispasr_audio_slice> crispasr_compute_vad_slices(const float* samples, int n_samples, int sample_rate,
                                                              const char* vad_model_path,
                                                              const crispasr_vad_options& opts) {
    std::vector<crispasr_audio_slice> slices;
    if (!vad_model_path || !*vad_model_path || n_samples <= 0)
        return slices;

    whisper_vad_context_params vcp = whisper_vad_default_context_params();
    vcp.n_threads = opts.n_threads;
    whisper_vad_context* vctx = whisper_vad_init_from_file_with_params(vad_model_path, vcp);
    if (!vctx) {
        fprintf(stderr, "crispasr: warning: failed to load VAD model '%s'\n", vad_model_path);
        return slices;
    }

    whisper_vad_params vp = whisper_vad_default_params();
    vp.threshold = opts.threshold;
    vp.min_speech_duration_ms = opts.min_speech_duration_ms;
    vp.min_silence_duration_ms = opts.min_silence_duration_ms;
    vp.speech_pad_ms = (float)opts.speech_pad_ms;

    whisper_vad_segments* vseg = whisper_vad_segments_from_samples(vctx, vp, samples, n_samples);
    const int nv = vseg ? whisper_vad_segments_n_segments(vseg) : 0;
    for (int i = 0; i < nv; i++) {
        // The whisper VAD API returns timestamps in centiseconds,
        // not seconds. Convert to seconds for sample index computation.
        const float t0_cs = whisper_vad_segments_get_segment_t0(vseg, i);
        const float t1_cs = whisper_vad_segments_get_segment_t1(vseg, i);
        const float t0s = t0_cs / 100.0f;
        const float t1s = t1_cs / 100.0f;
        const int s = std::max(0, (int)(t0s * sample_rate));
        const int e = std::min(n_samples, (int)(t1s * sample_rate));
        if (e > s) {
            slices.push_back({
                s,
                e,
                (int64_t)t0_cs,
                (int64_t)t1_cs,
            });
        }
    }
    if (vseg)
        whisper_vad_free_segments(vseg);
    whisper_vad_free(vctx);

    // Post-merge: combine adjacent VAD segments that are too short
    // or too close together. ASR models need at least a few seconds
    // of audio context to produce reliable output; tiny VAD segments
    // (< 2s) and short inter-segment gaps (< 1s) degrade quality.
    if (slices.size() > 1) {
        const int min_dur_samples = 3 * sample_rate;   // 3 s minimum
        const int merge_gap_samples = 1 * sample_rate; // merge if gap < 1 s
        std::vector<crispasr_audio_slice> merged;
        merged.push_back(slices[0]);
        for (size_t i = 1; i < slices.size(); i++) {
            auto& prev = merged.back();
            const int gap = slices[i].start - prev.end;
            const int prev_dur = prev.end - prev.start;
            if (gap < merge_gap_samples || prev_dur < min_dur_samples) {
                prev.end = slices[i].end;
                prev.t1_cs = slices[i].t1_cs;
            } else {
                merged.push_back(slices[i]);
            }
        }
        slices = std::move(merged);
    }

    // Post-split: break any VAD segment that exceeds chunk_seconds into
    // sub-segments. Prevents OOM on very long continuous speech (10+ min
    // lectures). We split into roughly equal parts.
    if (opts.chunk_seconds > 0) {
        const int max_samples = opts.chunk_seconds * sample_rate;
        std::vector<crispasr_audio_slice> split;
        for (auto& sl : slices) {
            const int dur = sl.end - sl.start;
            if (dur <= max_samples) {
                split.push_back(sl);
            } else {
                const int n_parts = (dur + max_samples - 1) / max_samples;
                const int part_samples = dur / n_parts;
                for (int p = 0; p < n_parts; p++) {
                    const int s = sl.start + p * part_samples;
                    const int e = (p == n_parts - 1) ? sl.end : sl.start + (p + 1) * part_samples;
                    split.push_back({
                        s,
                        e,
                        (int64_t)((double)s / sample_rate * 100.0),
                        (int64_t)((double)e / sample_rate * 100.0),
                    });
                }
            }
        }
        slices = std::move(split);
    }

    return slices;
}

std::vector<crispasr_audio_slice> crispasr_fixed_chunk_slices(int n_samples, int sample_rate, int chunk_seconds) {
    std::vector<crispasr_audio_slice> slices;
    if (n_samples <= 0)
        return slices;

    const int chunk_samples = chunk_seconds > 0 ? chunk_seconds * sample_rate : n_samples;
    if (n_samples <= chunk_samples) {
        const int64_t dur_cs = (int64_t)((double)n_samples / sample_rate * 100.0);
        slices.push_back({0, n_samples, 0, dur_cs});
        return slices;
    }
    for (int s = 0; s < n_samples; s += chunk_samples) {
        const int e = std::min(n_samples, s + chunk_samples);
        slices.push_back({
            s,
            e,
            (int64_t)((double)s / sample_rate * 100.0),
            (int64_t)((double)e / sample_rate * 100.0),
        });
    }
    return slices;
}

crispasr_stitched_audio crispasr_stitch_vad_slices(const float* samples, int /*n_samples*/, int sample_rate,
                                                   const std::vector<crispasr_audio_slice>& slices) {
    crispasr_stitched_audio result;
    if (slices.empty())
        return result;

    const int silence_samples = (int)(0.1f * sample_rate); // 0.1s silence gap

    size_t total = 0;
    for (const auto& sl : slices)
        total += (size_t)(sl.end - sl.start);
    total += (size_t)(slices.size() - 1) * silence_samples;

    result.samples.resize(total, 0.0f);
    result.mapping.reserve(slices.size() * 2);

    int offset = 0;
    for (size_t i = 0; i < slices.size(); i++) {
        const auto& sl = slices[i];
        const int seg_len = sl.end - sl.start;

        result.mapping.push_back({(int64_t)((double)offset / sample_rate * 100.0), sl.t0_cs});
        std::memcpy(result.samples.data() + offset, samples + sl.start, (size_t)seg_len * sizeof(float));
        offset += seg_len;
        result.mapping.push_back({(int64_t)((double)offset / sample_rate * 100.0), sl.t1_cs});

        if (i + 1 < slices.size())
            offset += silence_samples;
    }

    result.total_duration_cs = (int64_t)((double)offset / sample_rate * 100.0);
    return result;
}

int64_t crispasr_vad_remap_timestamp(const std::vector<crispasr_vad_mapping>& mapping, int64_t stitched_cs) {
    if (mapping.empty())
        return stitched_cs;
    if (stitched_cs <= mapping.front().stitched_cs)
        return mapping.front().original_cs;
    if (stitched_cs >= mapping.back().stitched_cs)
        return mapping.back().original_cs;

    size_t lo = 0, hi = mapping.size() - 1;
    while (lo + 1 < hi) {
        size_t mid = (lo + hi) / 2;
        if (mapping[mid].stitched_cs <= stitched_cs)
            lo = mid;
        else
            hi = mid;
    }
    const auto& a = mapping[lo];
    const auto& b = mapping[hi];
    if (b.stitched_cs == a.stitched_cs)
        return a.original_cs;
    const double frac = (double)(stitched_cs - a.stitched_cs) / (double)(b.stitched_cs - a.stitched_cs);
    return a.original_cs + (int64_t)(frac * (double)(b.original_cs - a.original_cs));
}
