// crisp_audio.h — model-agnostic Qwen-family audio tower for shared use by
// CrispASR (qwen3-asr) and CrispEmbed (BidirLM-Omni text+audio embedding).
//
// The audio tower is a Whisper-shape encoder used by Qwen-family multimodal
// models: 3 stride-2 Conv2D layers + sinusoidal positional embedding + N
// pre-LN encoder layers + final LN + 2-layer projection (proj1/GELU/proj2).
// Inputs are 16 kHz mono PCM (caller computes the log-mel spectrogram); the
// output is per-frame `output_dim`-dimensional features in the model's
// shared embedding space.
//
// Both consumers feed the same conv stem / encoder / projection architecture
// with different scalar params (d_model, n_window, output_dim) and slightly
// different chunking. Those scalars are read from the GGUF file the caller
// passes in, not hard-coded in this library.
//
// All functions are thread-unsafe per context — wrap with a mutex in
// multi-threaded callers.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct crisp_audio_context;

// Hyper-parameters that are not stored in GGUF metadata or that the caller
// wants to override at runtime.
struct crisp_audio_params {
    int n_threads;        // CPU thread count for ggml graph compute
    int verbosity;        // 0=silent 1=normal 2=verbose
    bool use_gpu;         // pick a GPU backend if available

    // Tensor-name prefix in the GGUF (e.g. "audio." for qwen3-asr,
    // "audio_tower." for BidirLM). All weight lookups are
    // `<prefix>conv2d1.weight` etc. Pass NULL to default to "audio.".
    const char* tensor_prefix;

    // Optional metadata-key prefix (e.g. "qwen3asr." or "bidirlm.audio.").
    // Used to read d_model/n_layers/etc. from GGUF. NULL = "crisp_audio.".
    const char* meta_prefix;
};

struct crisp_audio_params crisp_audio_params_default(void);

// Load the audio tower from a GGUF file.
//
// The GGUF must contain the audio-tower weights under `tensor_prefix` and
// scalar hparams (d_model, n_layers, n_heads, n_window, output_dim, …)
// either under `meta_prefix` or as part of a recognized model dialect.
//
// Returns NULL on failure. Caller must free with crisp_audio_free().
struct crisp_audio_context* crisp_audio_init_from_file(
    const char* gguf_path,
    const struct crisp_audio_params* params);

void crisp_audio_free(struct crisp_audio_context* ctx);

// Compute the log-mel spectrogram from raw 16 kHz mono float32 PCM, matching
// the HuggingFace WhisperFeatureExtractor recipe (n_fft=400, hop=160,
// n_mels=128, log10 + clip-to-8 + (x+4)/4 normalization).
//
// Returns a malloc'd float buffer of shape (n_mels, T_mel) row-major, or
// NULL on failure. Caller frees with free(). *out_n_mels and *out_T_mel
// are set on return.
float* crisp_audio_compute_mel(struct crisp_audio_context* ctx,
                               const float* samples, int n_samples,
                               int* out_n_mels, int* out_T_mel);

// Run the full audio tower forward: conv stem → pos embed → encoder layers
// → ln_post → proj1 → GELU → proj2.
//
// Input mel: (n_mels, T_mel) row-major float32, e.g. from compute_mel().
// Output: malloc'd float buffer of shape (N_total, output_dim) row-major,
// where N_total is the number of valid post-CNN frames after chunking
// (model-specific). *out_n_frames and *out_dim are set on return. Caller
// frees with free().
float* crisp_audio_encode(struct crisp_audio_context* ctx,
                          const float* mel, int n_mels, int T_mel,
                          int* out_n_frames, int* out_dim);

// Read scalar hparams of the loaded model. Useful for sanity checks at the
// caller and for sizing per-call buffers.
int crisp_audio_d_model(struct crisp_audio_context* ctx);
int crisp_audio_output_dim(struct crisp_audio_context* ctx);
int crisp_audio_n_layers(struct crisp_audio_context* ctx);
int crisp_audio_n_window(struct crisp_audio_context* ctx);

#ifdef __cplusplus
}
#endif
