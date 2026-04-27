#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct gemma4_e2b_context;

struct gemma4_e2b_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy
};

struct gemma4_e2b_context_params gemma4_e2b_context_default_params(void);

// Initialize from a GGUF file. Returns nullptr on failure.
struct gemma4_e2b_context* gemma4_e2b_init_from_file(const char* path_model, struct gemma4_e2b_context_params params);

// Transcribe PCM audio (16kHz mono float32). Returns malloc'd UTF-8 string (caller frees).
char* gemma4_e2b_transcribe(struct gemma4_e2b_context* ctx, const float* pcm, int n_samples);

// Free context and all associated memory.
void gemma4_e2b_free(struct gemma4_e2b_context* ctx);

// Set thread count after init.
void gemma4_e2b_set_n_threads(struct gemma4_e2b_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
