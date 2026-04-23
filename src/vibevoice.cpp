// vibevoice.cpp — Microsoft VibeVoice-ASR runtime.
//
// Pipeline: 24kHz PCM → acoustic σ-VAE encoder → connector → Qwen2 LM → text
//           24kHz PCM → semantic encoder → connector ↗
//
// Two parallel CNN tokenizer encoders (ConvNeXt blocks with depthwise conv),
// projected to LM space via FC connectors, then autoregressive Qwen2 decoder.

#include "vibevoice.h"
#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Hyperparams
// ===========================================================================

struct vibevoice_hparams {
    int d_lm = 1536;
    int n_lm_layers = 28;
    int n_heads = 12;
    int n_kv_heads = 2;
    int d_ffn = 8960;
    int vocab_size = 151936;
    int head_dim = 128;
    float rope_theta = 1000000.0f;
    int vae_dim_acoustic = 64;
    int vae_dim_semantic = 128;
    int n_encoder_stages = 7;
    int n_filters = 32;
    int total_downsample = 3200;
    std::vector<int> encoder_ratios;
    std::vector<int> encoder_depths;
};

// ===========================================================================
// Model
// ===========================================================================

struct vibevoice_model {
    vibevoice_hparams hp;
    std::map<std::string, ggml_tensor*> tensors;
};

struct vibevoice_context {
    vibevoice_model model;
    vibevoice_context_params params;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* weight_ctx = nullptr;
    // KV cache for LM decoder
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;
    std::vector<uint8_t> compute_meta;
};

// ===========================================================================
// Defaults
// ===========================================================================

extern "C" struct vibevoice_context_params vibevoice_context_default_params(void) {
    vibevoice_context_params p;
    p.n_threads = 4;
    p.max_new_tokens = 512;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

// ===========================================================================
// Init
// ===========================================================================

extern "C" struct vibevoice_context* vibevoice_init_from_file(const char* path_model,
                                                               struct vibevoice_context_params params) {
    auto* ctx = new vibevoice_context();
    ctx->params = params;
    auto& m = ctx->model;
    auto& hp = m.hp;

    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }

    hp.d_lm = core_gguf::kv_u32(gctx, "vibevoice.d_lm", 1536);
    hp.n_lm_layers = core_gguf::kv_u32(gctx, "vibevoice.n_lm_layers", 28);
    hp.n_heads = core_gguf::kv_u32(gctx, "vibevoice.n_heads", 12);
    hp.n_kv_heads = core_gguf::kv_u32(gctx, "vibevoice.n_kv_heads", 2);
    hp.d_ffn = core_gguf::kv_u32(gctx, "vibevoice.d_ffn", 8960);
    hp.vocab_size = core_gguf::kv_u32(gctx, "vibevoice.vocab_size", 151936);
    hp.head_dim = core_gguf::kv_u32(gctx, "vibevoice.head_dim", 128);
    hp.rope_theta = core_gguf::kv_f32(gctx, "vibevoice.rope_theta", 1000000.0f);
    hp.vae_dim_acoustic = core_gguf::kv_u32(gctx, "vibevoice.vae_dim_acoustic", 64);
    hp.vae_dim_semantic = core_gguf::kv_u32(gctx, "vibevoice.vae_dim_semantic", 128);
    hp.n_encoder_stages = core_gguf::kv_u32(gctx, "vibevoice.n_encoder_stages", 7);
    hp.n_filters = core_gguf::kv_u32(gctx, "vibevoice.n_filters", 32);
    hp.total_downsample = core_gguf::kv_u32(gctx, "vibevoice.total_downsample", 3200);

    // Read encoder arrays
    int ratios_key = gguf_find_key(gctx, "vibevoice.encoder_ratios");
    if (ratios_key >= 0) {
        int n = gguf_get_arr_n(gctx, ratios_key);
        hp.encoder_ratios.resize(n);
        for (int i = 0; i < n; i++)
            hp.encoder_ratios[i] = ((const int32_t*)gguf_get_arr_data(gctx, ratios_key))[i];
    }
    int depths_key = gguf_find_key(gctx, "vibevoice.encoder_depths");
    if (depths_key >= 0) {
        int n = gguf_get_arr_n(gctx, depths_key);
        hp.encoder_depths.resize(n);
        for (int i = 0; i < n; i++)
            hp.encoder_depths[i] = ((const int32_t*)gguf_get_arr_data(gctx, depths_key))[i];
    }

    gguf_free(gctx);

    if (params.verbosity >= 1) {
        fprintf(stderr, "vibevoice: d_lm=%d, layers=%d, heads=%d/%d, ffn=%d, vocab=%d\n", hp.d_lm, hp.n_lm_layers,
                hp.n_heads, hp.n_kv_heads, hp.d_ffn, hp.vocab_size);
        fprintf(stderr, "vibevoice: vae_acoustic=%d, vae_semantic=%d, downsample=%dx\n", hp.vae_dim_acoustic,
                hp.vae_dim_semantic, hp.total_downsample);
    }

    // Load weights
    ctx->backend = hp.d_lm > 0 ? (params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init())
                                : ggml_backend_cpu_init();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "vibevoice-asr", wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->weight_ctx = wl.ctx;
    ctx->buf = wl.buf;
    m.tensors = wl.tensors;

    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 65536, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(65536, false));

    if (params.verbosity >= 1)
        fprintf(stderr, "vibevoice: loaded %zu tensors\n", m.tensors.size());

    return ctx;
}

// ===========================================================================
// Free
// ===========================================================================

extern "C" void vibevoice_free(struct vibevoice_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->weight_ctx)
        ggml_free(ctx->weight_ctx);
    if (ctx->buf)
        ggml_backend_buffer_free(ctx->buf);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// Transcribe (stub — full implementation in next commit)
// ===========================================================================

extern "C" char* vibevoice_transcribe(struct vibevoice_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    auto& m = ctx->model;
    auto& hp = m.hp;

    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = m.tensors.find(name);
        return it != m.tensors.end() ? it->second : nullptr;
    };

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: %d samples (%.2fs at 24kHz)\n", n_samples, n_samples / 24000.0f);

    // TODO: implement full pipeline
    // 1. Build ggml graph for acoustic tokenizer encoder
    // 2. Build ggml graph for semantic tokenizer encoder
    // 3. Run connectors (FC1→RMSNorm→FC2)
    // 4. Build LM prefix and run autoregressive decoder

    fprintf(stderr, "vibevoice: runtime not yet implemented\n");
    return nullptr;
}
