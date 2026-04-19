// glm_asr.cpp — GLM-ASR-Nano runtime (zai-org/GLM-ASR-Nano-2512).
//
// Architecture: Whisper encoder (1280d, 32L, partial RoPE, LayerNorm+bias)
//             + 4-frame-stack projector (5120→4096,GELU → 4096→2048)
//             + Llama LLM (2048d, 28L, GQA 16/4, SwiGLU, RMSNorm)
//
// Closely follows voxtral.cpp — same building blocks, different sizes.

#include "glm_asr.h"

#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

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
#include <string>
#include <vector>

// ===========================================================================
// Model structures
// ===========================================================================

struct glm_asr_hparams {
    // Audio encoder
    int enc_hidden = 1280;
    int enc_n_layers = 32;
    int enc_n_heads = 20;
    int enc_n_kv_heads = 20;
    int enc_ff = 5120;
    int n_mels = 128;
    int enc_max_pos = 1500;
    float partial_rotary = 0.5f;
    // LLM decoder
    int llm_hidden = 2048;
    int llm_n_layers = 28;
    int llm_n_heads = 16;
    int llm_n_kv_heads = 4;
    int llm_ff = 6144;
    int llm_vocab = 59264;
    int llm_max_pos = 8192;
    float rms_eps = 1e-5f;
    // Special tokens
    int audio_token_id = 59260;
    int bos_token_id = 1;
    int eos_token_ids[4] = {59246, 59253, 59255, -1};
    int n_eos = 3;
    // Derived
    int enc_head_dim = 64;  // enc_hidden / enc_n_heads
    int llm_head_dim = 128; // llm_hidden / llm_n_heads
};

struct glm_enc_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
};

struct glm_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct glm_asr_model {
    glm_asr_hparams hp;

    // Audio encoder
    struct {
        ggml_tensor* conv1_w = nullptr;
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* conv2_w = nullptr;
        ggml_tensor* conv2_b = nullptr;
        ggml_tensor* ln_post_w = nullptr;
        ggml_tensor* ln_post_b = nullptr;
        std::vector<glm_enc_block> blocks;
    } audio;

    // Projector (4-frame stack → 2 linears)
    struct {
        ggml_tensor* linear1_w = nullptr;
        ggml_tensor* linear1_b = nullptr;
        ggml_tensor* linear2_w = nullptr;
        ggml_tensor* linear2_b = nullptr;
    } proj;

    // LLM
    struct {
        ggml_tensor* token_embd_w = nullptr;
        ggml_tensor* output_norm_w = nullptr;
        ggml_tensor* lm_head_w = nullptr;
        std::vector<glm_llm_block> blocks;
    } llm;

    // Tokenizer
    std::vector<std::string> vocab;

    // GGUF context (owns the weight memory)
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct glm_asr_context {
    glm_asr_context_params params;
    glm_asr_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;

    int n_threads = 4;
};

// ===========================================================================
// Implementation
// ===========================================================================

extern "C" struct glm_asr_context_params glm_asr_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1};
}

extern "C" struct glm_asr_context* glm_asr_init_from_file(const char* path_model,
                                                          struct glm_asr_context_params params) {
    auto* ctx = new glm_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    // Load GGUF
    auto& m = ctx->model;
    auto& hp = m.hp;

    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path_model);
        if (!gctx) {
            fprintf(stderr, "glm_asr: failed to open '%s'\n", path_model);
            delete ctx;
            return nullptr;
        }
        hp.enc_hidden = core_gguf::kv_u32(gctx, "glmasr.audio.hidden_size", hp.enc_hidden);
        hp.enc_n_layers = core_gguf::kv_u32(gctx, "glmasr.audio.num_layers", hp.enc_n_layers);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "glmasr.audio.num_heads", hp.enc_n_heads);
        hp.enc_n_kv_heads = core_gguf::kv_u32(gctx, "glmasr.audio.num_kv_heads", hp.enc_n_kv_heads);
        hp.enc_ff = core_gguf::kv_u32(gctx, "glmasr.audio.intermediate_size", hp.enc_ff);
        hp.n_mels = core_gguf::kv_u32(gctx, "glmasr.audio.num_mel_bins", hp.n_mels);
        hp.enc_max_pos = core_gguf::kv_u32(gctx, "glmasr.audio.max_position_embeddings", hp.enc_max_pos);
        hp.partial_rotary = core_gguf::kv_f32(gctx, "glmasr.audio.partial_rotary_factor", hp.partial_rotary);

        hp.llm_hidden = core_gguf::kv_u32(gctx, "glmasr.llm.hidden_size", hp.llm_hidden);
        hp.llm_n_layers = core_gguf::kv_u32(gctx, "glmasr.llm.num_layers", hp.llm_n_layers);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "glmasr.llm.num_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "glmasr.llm.num_kv_heads", hp.llm_n_kv_heads);
        hp.llm_ff = core_gguf::kv_u32(gctx, "glmasr.llm.intermediate_size", hp.llm_ff);
        hp.llm_vocab = core_gguf::kv_u32(gctx, "glmasr.llm.vocab_size", hp.llm_vocab);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "glmasr.llm.max_position_embeddings", hp.llm_max_pos);
        hp.rms_eps = core_gguf::kv_f32(gctx, "glmasr.llm.rms_norm_eps", hp.rms_eps);

        hp.audio_token_id = core_gguf::kv_u32(gctx, "glmasr.audio_token_id", hp.audio_token_id);
        hp.bos_token_id = core_gguf::kv_u32(gctx, "glmasr.bos_token_id", hp.bos_token_id);
        hp.n_eos = core_gguf::kv_u32(gctx, "glmasr.num_eos_tokens", hp.n_eos);
        for (int i = 0; i < hp.n_eos && i < 4; i++) {
            char key[64];
            snprintf(key, sizeof(key), "glmasr.eos_token_id_%d", i);
            hp.eos_token_ids[i] = core_gguf::kv_u32(gctx, key, hp.eos_token_ids[i]);
        }

        hp.enc_head_dim = hp.enc_hidden / hp.enc_n_heads;
        hp.llm_head_dim = hp.llm_hidden / hp.llm_n_heads;

        // Tokenizer
        m.vocab.resize(hp.llm_vocab);
        const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            const int n = gguf_get_arr_n(gctx, tok_key);
            for (int i = 0; i < n && i < hp.llm_vocab; i++) {
                const char* s = gguf_get_arr_str(gctx, tok_key, i);
                if (s)
                    m.vocab[i] = s;
            }
        }

        gguf_free(gctx);
    }

    // ---- pass 2: load tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "glm_asr", wl)) {
        fprintf(stderr, "glm_asr: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;

    // Map tensors
    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        if (it == wl.tensors.end()) {
            fprintf(stderr, "glm_asr: tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };
    auto try_get = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    // Audio encoder tensors
    m.audio.conv1_w = get("audio.conv1.weight");
    m.audio.conv1_b = get("audio.conv1.bias");
    m.audio.conv2_w = get("audio.conv2.weight");
    m.audio.conv2_b = get("audio.conv2.bias");
    m.audio.ln_post_w = get("audio.norm.weight");
    m.audio.ln_post_b = try_get("audio.norm.bias");
    m.audio.blocks.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        char buf[128];
        auto& b = m.audio.blocks[i];
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_norm.weight", i);
        b.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_norm.bias", i);
        b.attn_norm_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_q.weight", i);
        b.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_q.bias", i);
        b.attn_q_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_k.weight", i);
        b.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_v.weight", i);
        b.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_out.weight", i);
        b.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_out.bias", i);
        b.attn_out_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn_norm.weight", i);
        b.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn_norm.bias", i);
        b.ffn_norm_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.up.weight", i);
        b.ffn_up_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.up.bias", i);
        b.ffn_up_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.down.weight", i);
        b.ffn_down_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.down.bias", i);
        b.ffn_down_b = try_get(buf);
    }

    // Projector
    m.proj.linear1_w = get("proj.linear_1.weight");
    m.proj.linear1_b = try_get("proj.linear_1.bias");
    m.proj.linear2_w = get("proj.linear_2.weight");
    m.proj.linear2_b = try_get("proj.linear_2.bias");

    // LLM tensors
    m.llm.token_embd_w = get("llm.token_embd.weight");
    m.llm.output_norm_w = get("llm.norm.weight");
    m.llm.lm_head_w = get("lm_head.weight");
    m.llm.blocks.resize(hp.llm_n_layers);
    for (int i = 0; i < hp.llm_n_layers; i++) {
        char buf[128];
        auto& b = m.llm.blocks[i];
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_norm.weight", i);
        b.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn_norm.weight", i);
        b.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_q.weight", i);
        b.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_k.weight", i);
        b.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_v.weight", i);
        b.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_out.weight", i);
        b.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.gate.weight", i);
        b.ffn_gate_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.up.weight", i);
        b.ffn_up_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.down.weight", i);
        b.ffn_down_w = get(buf);
    }

    // Scheduler
    int n_be = 1;
    ggml_backend_t backends[2] = {ctx->backend, nullptr};
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
        backends[n_be++] = ctx->backend_cpu;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    int n_audio_t = 0, n_llm_t = 0;
    for (int i = 0; i < hp.enc_n_layers; i++) {
        if (m.audio.blocks[i].attn_q_w)
            n_audio_t++;
    }
    for (int i = 0; i < hp.llm_n_layers; i++) {
        if (m.llm.blocks[i].attn_q_w)
            n_llm_t++;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "glm_asr: loaded %d audio + %d LLM layers, vocab %d\n", n_audio_t, n_llm_t, hp.llm_vocab);
    }

    return ctx;
}

extern "C" void glm_asr_free(struct glm_asr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" const char* glm_asr_token_text(struct glm_asr_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->model.vocab.size())
        return "";
    return ctx->model.vocab[id].c_str();
}

extern "C" int32_t* glm_asr_tokenize(struct glm_asr_context* ctx, const char* text, int* out_n_tokens) {
    // Simple tokenizer: look up each known special token, then byte-fallback.
    // For a production implementation, this would use the full tokenizer.
    if (!ctx || !text || !out_n_tokens)
        return nullptr;

    std::vector<int32_t> ids;
    std::string s(text);

    // Map special tokens
    struct {
        const char* text;
        int id;
    } specials[] = {
        {"<|begin_of_audio|>", 59261},
        {"<|end_of_audio|>", 59262},
        {"<|pad|>", 59260},
        {"<|user|>", 59253},
        {"<|assistant|>", 59254},
        {"<|system|>", 59252},
        {"<|endoftext|>", 59246},
        {"\n", -1}, // handled below
    };

    size_t pos = 0;
    while (pos < s.size()) {
        bool found = false;
        for (const auto& sp : specials) {
            size_t len = strlen(sp.text);
            if (s.compare(pos, len, sp.text) == 0) {
                if (sp.id >= 0)
                    ids.push_back(sp.id);
                pos += len;
                found = true;
                break;
            }
        }
        if (!found) {
            // Byte fallback — find the vocab entry for this character/substring
            // For now, skip unknown bytes
            pos++;
        }
    }

    *out_n_tokens = (int)ids.size();
    if (ids.empty())
        return nullptr;
    auto* result = (int32_t*)malloc(ids.size() * sizeof(int32_t));
    memcpy(result, ids.data(), ids.size() * sizeof(int32_t));
    return result;
}

// Stub implementations — the full pipeline will be implemented iteratively
// following the voxtral.cpp pattern. For now, expose the API surface.

extern "C" char* glm_asr_transcribe(struct glm_asr_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    // TODO: implement full pipeline (mel → encoder → projector → prompt → LLM decode)
    fprintf(stderr, "glm_asr: transcribe() not yet implemented\n");
    return strdup("");
}

extern "C" float* glm_asr_compute_mel(struct glm_asr_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                      int* out_T_mel) {
    if (!ctx || !samples)
        return nullptr;
    // Use core_mel with whisper params (same as GLM-ASR's feature extractor)
    // TODO: GLM-ASR mel extraction (same params as whisper large-v3).
    // Need to bake mel_filters + window into the GGUF (like voxtral does),
    // or generate them at runtime.
    (void)n_samples;
    (void)out_n_mels;
    (void)out_T_mel;
    fprintf(stderr, "glm_asr: compute_mel() not yet implemented\n");
    return nullptr;
}

extern "C" float* glm_asr_embed_tokens(struct glm_asr_context* ctx, const int32_t* ids, int n_ids) {
    if (!ctx || !ids || n_ids <= 0)
        return nullptr;

    const auto& m = ctx->model;
    const int d = m.hp.llm_hidden;

    // Build tiny graph: token_embd lookup
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ids);
    ggml_set_name(inp, "token_ids");
    ggml_set_input(inp);

    ggml_tensor* emb = ggml_get_rows(ctx0, m.llm.token_embd_w, inp);
    ggml_set_name(emb, "embeddings");
    ggml_build_forward_expand(gf, emb);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "token_ids"), ids, 0, n_ids * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeddings");
    float* result = (float*)malloc((size_t)n_ids * d * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)n_ids * d * sizeof(float));
    return result;
}

extern "C" bool glm_asr_kv_init(struct glm_asr_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    const auto& hp = ctx->model.hp;
    const int hd = hp.llm_head_dim;
    const int n_kv = hp.llm_n_kv_heads;
    const int nl = hp.llm_n_layers;
    size_t k_size = (size_t)ggml_type_size(GGML_TYPE_F16) * hd * max_ctx * n_kv * nl;
    size_t total = k_size * 2 + ggml_tensor_overhead() * 2;

    ggml_init_params ip = {total, nullptr, false};
    ctx->kv_ctx = ggml_init(ip);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }
    glm_asr_kv_reset(ctx);
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "glm_asr: kv cache %zu MiB\n", (k_size * 2) >> 20);
    }
    return true;
}

extern "C" void glm_asr_kv_reset(struct glm_asr_context* ctx) {
    if (!ctx || !ctx->kv_buf)
        return;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
}

extern "C" float* glm_asr_run_encoder(struct glm_asr_context* ctx, const float* mel, int n_mels, int T_mel, int* out_N,
                                      int* out_dim) {
    // TODO: build encoder graph (whisper-style + partial RoPE)
    // For now, stub
    fprintf(stderr, "glm_asr: run_encoder() not yet implemented\n");
    return nullptr;
}

extern "C" float* glm_asr_run_llm_kv(struct glm_asr_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                     int* out_n_tokens, int* out_vocab_size) {
    // TODO: build LLM graph (Llama-style with GQA)
    // For now, stub
    fprintf(stderr, "glm_asr: run_llm_kv() not yet implemented\n");
    return nullptr;
}
