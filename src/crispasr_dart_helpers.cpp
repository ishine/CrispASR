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
// Version reporting for the Dart binding
// =========================================================================

CA_EXPORT const char * crispasr_dart_helpers_version(void) {
    return "0.2.0";
}
