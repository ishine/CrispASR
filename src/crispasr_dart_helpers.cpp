// CrispASR — C-ABI helpers used by the Dart FFI binding in
// `flutter/crispasr/`. These wrap the handful of whisper.h entry points that
// Dart can't call directly: functions that take or return structs by value,
// plus a couple of convenience wrappers that would otherwise need Dart to
// mirror the full `whisper_full_params` / `whisper_token_data` layouts.
//
// Every symbol here is plain C linkage, prefixed `crispasr_` so it can't
// collide with upstream whisper.h identifiers. Keep signatures stable once
// published — these are part of `package:crispasr`'s ABI contract.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "whisper.h"

#ifdef _WIN32
  #define CA_EXPORT extern "C" __declspec(dllexport)
#else
  #define CA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// =========================================================================
// whisper_full_params setters
// =========================================================================
//
// Dart holds the params as an opaque `Pointer<Void>` returned by
// `whisper_full_default_params_by_ref`. Rather than mirror the struct layout
// (~40 fields, volatile across upstream bumps), we expose a setter per field
// we actually care about.

CA_EXPORT void crispasr_params_set_language(whisper_full_params * p,
                                            const char *          lang) {
    if (p) p->language = lang; // caller must keep the string alive
}
CA_EXPORT void crispasr_params_set_translate(whisper_full_params * p, int v) {
    if (p) p->translate = v != 0;
}
CA_EXPORT void crispasr_params_set_detect_language(whisper_full_params * p, int v) {
    if (p) p->detect_language = v != 0;
}
CA_EXPORT void crispasr_params_set_token_timestamps(whisper_full_params * p, int v) {
    if (p) p->token_timestamps = v != 0;
}
CA_EXPORT void crispasr_params_set_n_threads(whisper_full_params * p, int n) {
    if (p) p->n_threads = n;
}
CA_EXPORT void crispasr_params_set_max_len(whisper_full_params * p, int n) {
    if (p) p->max_len = n;
}
CA_EXPORT void crispasr_params_set_split_on_word(whisper_full_params * p, int v) {
    if (p) p->split_on_word = v != 0;
}
CA_EXPORT void crispasr_params_set_no_context(whisper_full_params * p, int v) {
    if (p) p->no_context = v != 0;
}
CA_EXPORT void crispasr_params_set_single_segment(whisper_full_params * p, int v) {
    if (p) p->single_segment = v != 0;
}
CA_EXPORT void crispasr_params_set_print_realtime(whisper_full_params * p, int v) {
    if (p) p->print_realtime = v != 0;
}
CA_EXPORT void crispasr_params_set_print_progress(whisper_full_params * p, int v) {
    if (p) p->print_progress = v != 0;
}
CA_EXPORT void crispasr_params_set_print_timestamps(whisper_full_params * p, int v) {
    if (p) p->print_timestamps = v != 0;
}
CA_EXPORT void crispasr_params_set_print_special(whisper_full_params * p, int v) {
    if (p) p->print_special = v != 0;
}
CA_EXPORT void crispasr_params_set_suppress_blank(whisper_full_params * p, int v) {
    if (p) p->suppress_blank = v != 0;
}
CA_EXPORT void crispasr_params_set_temperature(whisper_full_params * p, float t) {
    if (p) p->temperature = t;
}
CA_EXPORT void crispasr_params_set_initial_prompt(whisper_full_params * p,
                                                  const char *          prompt) {
    if (p) p->initial_prompt = prompt; // caller owns the string
}

// =========================================================================
// Token-level timestamp getters
// =========================================================================
//
// `whisper_full_get_token_data` returns a `whisper_token_data` *by value*,
// which Dart FFI can't handle portably. Expose each field we need as a
// scalar-returning helper.

CA_EXPORT int64_t crispasr_token_t0(whisper_context * ctx, int i_seg, int i_tok) {
    if (!ctx) return 0;
    return whisper_full_get_token_data(ctx, i_seg, i_tok).t0;
}
CA_EXPORT int64_t crispasr_token_t1(whisper_context * ctx, int i_seg, int i_tok) {
    if (!ctx) return 0;
    return whisper_full_get_token_data(ctx, i_seg, i_tok).t1;
}
CA_EXPORT float crispasr_token_p(whisper_context * ctx, int i_seg, int i_tok) {
    if (!ctx) return 0.0f;
    return whisper_full_get_token_data(ctx, i_seg, i_tok).p;
}

// =========================================================================
// Language detection
// =========================================================================
//
// Mel + encode + `whisper_lang_auto_detect`. Writes the ISO-639 code into
// `out_code` (e.g. "de") and returns the detected-language probability.
// Returns negative on error.

CA_EXPORT float crispasr_detect_language(whisper_context * ctx,
                                         const float *     pcm,
                                         int               n_samples,
                                         int               n_threads,
                                         char *            out_code,
                                         int               out_cap) {
    if (!ctx || !pcm || n_samples <= 0 || !out_code || out_cap <= 0) {
        return -1.0f;
    }

    // whisper requires mel + encode before lang auto-detect can run.
    if (whisper_pcm_to_mel(ctx, pcm, n_samples, n_threads > 0 ? n_threads : 4) != 0) {
        return -2.0f;
    }
    if (whisper_encode(ctx, 0, n_threads > 0 ? n_threads : 4) != 0) {
        return -3.0f;
    }

    std::vector<float> probs(whisper_lang_max_id() + 1, 0.0f);
    const int lang_id = whisper_lang_auto_detect(ctx, 0, n_threads > 0 ? n_threads : 4, probs.data());
    if (lang_id < 0) return -4.0f;

    const char * code = whisper_lang_str(lang_id);
    if (!code) return -5.0f;

    std::strncpy(out_code, code, out_cap - 1);
    out_code[out_cap - 1] = '\0';
    return probs[lang_id];
}

// =========================================================================
// VAD — run Silero on PCM, return [start_s, end_s] pairs
// =========================================================================
//
// `out_spans` is a malloc'd array of floats (2 per span). The caller must
// pass the pointer back to `crispasr_vad_free` when done. Returns the number
// of speech segments detected (>= 0), or a negative error.
//
//   -1  bad arguments
//   -2  model init failed
//   -3  VAD inference failed

CA_EXPORT int crispasr_vad_segments(const char *  vad_model_path,
                                    const float * pcm,
                                    int           n_samples,
                                    int           sample_rate,
                                    float         threshold,
                                    int           min_speech_ms,
                                    int           min_silence_ms,
                                    int           n_threads,
                                    bool          use_gpu,
                                    float **      out_spans) {
    if (!vad_model_path || !pcm || n_samples <= 0 || !out_spans) return -1;
    *out_spans = nullptr;

    whisper_vad_context_params cparams = whisper_vad_default_context_params();
    cparams.n_threads  = n_threads > 0 ? n_threads : 4;
    cparams.use_gpu    = use_gpu;
    cparams.gpu_device = 0;

    whisper_vad_context * vctx =
        whisper_vad_init_from_file_with_params(vad_model_path, cparams);
    if (!vctx) return -2;

    whisper_vad_params vparams = whisper_vad_default_params();
    if (threshold      > 0.0f) vparams.threshold               = threshold;
    if (min_speech_ms  > 0)    vparams.min_speech_duration_ms  = min_speech_ms;
    if (min_silence_ms > 0)    vparams.min_silence_duration_ms = min_silence_ms;

    whisper_vad_segments * segs =
        whisper_vad_segments_from_samples(vctx, vparams, pcm, n_samples);
    if (!segs) {
        whisper_vad_free(vctx);
        return -3;
    }

    const int n = whisper_vad_segments_n_segments(segs);
    if (n > 0) {
        float * buf = (float *) std::malloc(sizeof(float) * 2 * n);
        if (!buf) {
            whisper_vad_free_segments(segs);
            whisper_vad_free(vctx);
            return -2;
        }
        for (int i = 0; i < n; ++i) {
            buf[2 * i + 0] = whisper_vad_segments_get_segment_t0(segs, i);
            buf[2 * i + 1] = whisper_vad_segments_get_segment_t1(segs, i);
        }
        *out_spans = buf;
    }

    whisper_vad_free_segments(segs);
    whisper_vad_free(vctx);
    // `sample_rate` parameter is accepted for API future-proofing even though
    // Silero VAD internally assumes 16 kHz — if we later add automatic
    // resampling here, callers don't have to change.
    (void) sample_rate;
    return n;
}

CA_EXPORT void crispasr_vad_free(float * spans) {
    if (spans) std::free(spans);
}

// =========================================================================
// Streaming transcription
// =========================================================================
//
// Port of `examples/stream/stream.cpp`'s rolling-window approach, but
// packaged as a pure-C-ABI struct so Dart can drive it without spinning
// its own threads. Non-blocking: caller feeds PCM in chunks of any size,
// and each feed whose accumulation crosses `step_ms` runs a single
// `whisper_full` on the last `length_ms` of audio (plus a small `keep_ms`
// context carry-over) and returns the concatenated text.
//
// This is the same "sliding-window" trick the CLI uses. It is not true
// token-level streaming — it is chunked batch with context carry, which
// is what whisper.cpp itself supports.

struct crispasr_stream {
    whisper_context * ctx = nullptr;      // not owned
    int n_threads         = 4;
    int step_ms           = 3000;
    int length_ms         = 10000;
    int keep_ms           = 200;
    std::string language;                 // empty = auto
    bool translate        = false;

    int n_samples_step    = 0;            // cached from step_ms
    int n_samples_length  = 0;
    int n_samples_keep    = 0;

    std::vector<float> accum;             // samples fed since last decode
    std::vector<float> history;           // last decoded window (for carry)

    // Last decode output, held here until caller pulls it with
    // `crispasr_stream_get_text`.
    std::string out_text;
    double out_t0_s       = 0.0;
    double out_t1_s       = 0.0;
    bool   has_output     = false;

    // Monotonic counter so callers can detect when output has been replaced
    // by a subsequent decode even if the text didn't visibly change.
    int64_t decode_counter = 0;

    double stream_time_s  = 0.0;          // total audio fed, in seconds
};

CA_EXPORT crispasr_stream * crispasr_stream_open(whisper_context * ctx,
                                                 int               n_threads,
                                                 int               step_ms,
                                                 int               length_ms,
                                                 int               keep_ms,
                                                 const char *      language,
                                                 int               translate) {
    if (!ctx) return nullptr;
    auto * s = new crispasr_stream();
    s->ctx       = ctx;
    s->n_threads = n_threads > 0 ? n_threads : 4;
    s->step_ms   = step_ms   > 0 ? step_ms   : 3000;
    s->length_ms = length_ms > 0 ? length_ms : 10000;
    s->keep_ms   = keep_ms   >= 0 ? keep_ms  : 200;
    s->translate = translate != 0;
    if (language && language[0] != '\0') s->language = language;

    constexpr int kSampleRate = 16000;
    s->n_samples_step   = (int) (1e-3 * s->step_ms   * kSampleRate);
    s->n_samples_length = (int) (1e-3 * s->length_ms * kSampleRate);
    s->n_samples_keep   = (int) (1e-3 * s->keep_ms   * kSampleRate);
    return s;
}

CA_EXPORT void crispasr_stream_close(crispasr_stream * s) {
    if (!s) return;
    delete s;
}

static int crispasr_stream_run_decode(crispasr_stream * s) {
    // Assemble the decode window: tail of `history` (length `n_samples_take`)
    // + all of `accum`.
    const int n_new  = (int) s->accum.size();
    const int n_take = std::min((int) s->history.size(),
                                std::max(0, s->n_samples_keep + s->n_samples_length - n_new));

    std::vector<float> pcm;
    pcm.reserve(n_take + n_new);
    if (n_take > 0) {
        const size_t start = s->history.size() - (size_t) n_take;
        pcm.insert(pcm.end(),
                   s->history.begin() + start,
                   s->history.end());
    }
    pcm.insert(pcm.end(), s->accum.begin(), s->accum.end());

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;
    wparams.single_segment   = true;         // mirror stream.cpp non-VAD path
    wparams.no_timestamps    = false;
    wparams.translate        = s->translate;
    wparams.n_threads        = s->n_threads;
    wparams.language         = s->language.empty() ? nullptr : s->language.c_str();
    wparams.detect_language  = s->language.empty();
    wparams.no_context       = true;

    if (whisper_full(s->ctx, wparams, pcm.data(), (int) pcm.size()) != 0) {
        return -1;
    }

    // Concatenate all segments produced by this decode.
    const int n_seg = whisper_full_n_segments(s->ctx);
    std::string text;
    double t0_s = 1e18;
    double t1_s = 0.0;
    for (int i = 0; i < n_seg; ++i) {
        const char * segtext = whisper_full_get_segment_text(s->ctx, i);
        if (segtext) text += segtext;

        const double t0 = whisper_full_get_segment_t0(s->ctx, i) / 100.0;
        const double t1 = whisper_full_get_segment_t1(s->ctx, i) / 100.0;
        if (t0 < t0_s) t0_s = t0;
        if (t1 > t1_s) t1_s = t1;
    }
    if (n_seg == 0) { t0_s = 0.0; t1_s = 0.0; }

    // Re-base timestamps onto absolute stream time: the last sample fed
    // sits at `stream_time_s`; the start of the decode window sits
    // `pcm.size() / 16000` seconds before that.
    const double win_end_abs   = s->stream_time_s;
    const double win_start_abs = win_end_abs - (double) pcm.size() / 16000.0;
    s->out_text  = std::move(text);
    s->out_t0_s  = win_start_abs + t0_s;
    s->out_t1_s  = win_start_abs + t1_s;
    s->has_output = true;
    s->decode_counter += 1;

    // Keep the last ~`length_ms + keep_ms` of audio as history so the next
    // decode can carry context. Anything older is dropped.
    s->history = pcm;
    const int max_hist = s->n_samples_length + s->n_samples_keep;
    if ((int) s->history.size() > max_hist) {
        s->history.erase(s->history.begin(),
                         s->history.begin() + ((int) s->history.size() - max_hist));
    }
    s->accum.clear();
    return 0;
}

CA_EXPORT int crispasr_stream_feed(crispasr_stream * s,
                                   const float *     pcm,
                                   int               n_samples) {
    if (!s || !pcm || n_samples <= 0) return -1;
    s->accum.insert(s->accum.end(), pcm, pcm + n_samples);
    s->stream_time_s += (double) n_samples / 16000.0;

    if ((int) s->accum.size() < s->n_samples_step) {
        return 0; // still buffering
    }

    if (crispasr_stream_run_decode(s) != 0) return -2;
    return 1; // new output ready
}

CA_EXPORT int crispasr_stream_get_text(crispasr_stream * s,
                                       char *            out_text,
                                       int               out_cap,
                                       double *          out_t0_s,
                                       double *          out_t1_s,
                                       int64_t *         out_counter) {
    if (!s || !out_text || out_cap <= 0) return -1;
    if (!s->has_output) {
        out_text[0] = '\0';
        if (out_t0_s) *out_t0_s = 0.0;
        if (out_t1_s) *out_t1_s = 0.0;
        if (out_counter) *out_counter = 0;
        return 0;
    }
    std::strncpy(out_text, s->out_text.c_str(), out_cap - 1);
    out_text[out_cap - 1] = '\0';
    if (out_t0_s) *out_t0_s = s->out_t0_s;
    if (out_t1_s) *out_t1_s = s->out_t1_s;
    if (out_counter) *out_counter = s->decode_counter;
    return (int) s->out_text.size();
}

/// Force a decode on whatever audio is currently buffered, regardless of
/// whether we hit the step threshold. Useful when the caller knows the
/// audio has ended and wants a final flush.
CA_EXPORT int crispasr_stream_flush(crispasr_stream * s) {
    if (!s) return -1;
    if (s->accum.empty()) return 0;
    return crispasr_stream_run_decode(s) == 0 ? 1 : -2;
}

// =========================================================================
// Version reporting for the Dart binding
// =========================================================================

CA_EXPORT const char * crispasr_dart_helpers_version(void) {
    return "0.3.0";
}
