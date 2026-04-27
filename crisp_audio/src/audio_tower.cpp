// audio_tower.cpp — shared Qwen-family audio-tower implementation.
//
// Extracted from CrispASR's qwen3_asr.cpp (the Qwen3-ASR audio encoder) and
// generalized so the same code can serve BidirLM-Omni (CrispEmbed) and
// future Qwen-family multimodal models that share this architecture.
//
// Architecture:
//   PCM → log-mel (128 bins) → 3× Conv2D(stride=2,GELU) → conv_out linear
//   → +sinusoidal_pos → N × pre-LN encoder layers (MHA + GELU FFN, both
//   pre-LN) → ln_post → proj1 → GELU → proj2 → (n_frames, output_dim)
//
// Per-model scalars (d_model, n_layers, n_heads, n_window, output_dim,
// etc.) are read from the GGUF file at load time — nothing here is
// hard-coded for a specific model.
//
// This file is intentionally a stub at extraction-start. The real
// implementation will be lifted from ../src/qwen3_asr.cpp in a follow-up
// commit, with d_model and friends parameterized.

#include "crisp_audio.h"

#include <cstdio>
#include <cstdlib>

extern "C" {

struct crisp_audio_context {
    // TODO: model + ggml backend + cached graphs go here when the
    // extraction lands. See ../src/qwen3_asr.cpp:154-200 for the full
    // shape that needs to come over.
    int placeholder;
};

struct crisp_audio_params crisp_audio_params_default(void) {
    struct crisp_audio_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.tensor_prefix = nullptr;  // → "audio."
    p.meta_prefix = nullptr;    // → "crisp_audio."
    return p;
}

struct crisp_audio_context* crisp_audio_init_from_file(
    const char* /*gguf_path*/,
    const struct crisp_audio_params* /*params*/) {
    std::fprintf(stderr, "crisp_audio: stub — implementation pending extraction\n");
    return nullptr;
}

void crisp_audio_free(struct crisp_audio_context* ctx) {
    delete ctx;
}

float* crisp_audio_compute_mel(struct crisp_audio_context* /*ctx*/,
                               const float* /*samples*/, int /*n_samples*/,
                               int* /*out_n_mels*/, int* /*out_T_mel*/) {
    return nullptr;
}

float* crisp_audio_encode(struct crisp_audio_context* /*ctx*/,
                          const float* /*mel*/, int /*n_mels*/, int /*T_mel*/,
                          int* /*out_n_frames*/, int* /*out_dim*/) {
    return nullptr;
}

int crisp_audio_d_model(struct crisp_audio_context*) { return 0; }
int crisp_audio_output_dim(struct crisp_audio_context*) { return 0; }
int crisp_audio_n_layers(struct crisp_audio_context*) { return 0; }
int crisp_audio_n_window(struct crisp_audio_context*) { return 0; }

}  // extern "C"
