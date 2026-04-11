// src/core/attention.h — shared multi-head attention helpers (header-only).
//
// Replaces the Q/K/V-projection + reshape + RoPE + GQA-expand + flash-attn +
// output-projection block that every LLM-based model in src/ has 1–2 copies
// of. The helper is header-only so the compiler inlines it straight into
// each caller, producing the exact same ggml op sequence as the original
// inline code and preserving bit-identical graph execution.
//
// Scope of the initial version (this commit):
//
//   core_attn::llama_self_attn_kv()  — the classic Llama / Mistral LLM
//     attention block: RMSNorm weights applied by caller, no biases on
//     Q/K/V/O, NEOX RoPE, optional GQA expansion, ggml_flash_attn_ext with
//     a caller-supplied causal-or-sliding-window mask, reshape + output
//     projection. Used by voxtral, voxtral4b, qwen3 (without Q/K norm),
//     and granite LLM decoders.
//
// Follow-up variants (to be added when their first consumer migrates):
//
//   * post-projection Q/K RMSNorm (qwen3)
//   * separate audio-encoder variant with biases + no RoPE (voxtral audio)
//   * adaptive scale / residual_multiplier (granite µP)
//   * KV-cache lookup that returns a (K, V) pair instead of taking
//     pre-permuted inputs (needed when KV cache is stored in a different
//     layout than (head_dim, T, n_heads))
//
// This staged approach keeps the first commit narrow and verifiable. Every
// new caller either fits the existing helper or adds a new sibling helper
// with its own regression test.

#pragma once

#include "ggml.h"

namespace core_attn {

// Parameters that differ from call to call. Everything here is a plain
// value type so the compiler can inline the caller's constants into the
// helper's ggml_* op chain.
struct LlamaSelfAttnParams {
    int   n_heads;        // query heads (== n_kv_heads for MHA)
    int   n_kv_heads;     // key/value heads; with GQA, n_heads / n_kv_heads > 1
    int   head_dim;       // per-head dimension
    int   n_kv_grp;       // == n_heads / n_kv_heads (caller precomputes)
    int   n_ctx_orig;     // rope n_ctx_orig (usually llm_max_pos)
    float rope_theta;
    float attn_scale;     // usually 1 / sqrt(head_dim); pass explicitly
};

// Llama / Mistral-style self-attention with optional GQA, NEOX RoPE, and
// ggml_flash_attn_ext.
//
// Inputs:
//   x              [d_model, T]  — RMSNormed input for this layer (the
//                                   caller does the norm + the learned
//                                   scale multiplication)
//   q_w,k_w,v_w    Q/K/V weight tensors (no biases in the LLM case)
//   o_w            output projection weight
//   positions      [T]           — RoPE position ids
//   mask           [ctx, T] F16  — causal / sliding-window mask or nullptr
//                                   for no-mask (voxtral audio case)
//
// Output:
//   attn           [d_model, T]  — post-output-projection tensor. The
//                                   caller adds it to the residual.
static inline ggml_tensor * llama_self_attn(
    ggml_context * ctx,
    ggml_tensor  * x,
    ggml_tensor  * q_w,
    ggml_tensor  * k_w,
    ggml_tensor  * v_w,
    ggml_tensor  * o_w,
    ggml_tensor  * positions,
    ggml_tensor  * mask,
    const LlamaSelfAttnParams & p)
{
    const int hd      = p.head_dim;
    const int n_q     = p.n_heads;
    const int n_kv    = p.n_kv_heads;
    const int n_ctx   = p.n_ctx_orig;
    const int grp     = p.n_kv_grp;

    // Q/K/V projections (no biases for the Llama case).
    ggml_tensor * Q = ggml_mul_mat(ctx, q_w, x);
    ggml_tensor * K = ggml_mul_mat(ctx, k_w, x);
    ggml_tensor * V = ggml_mul_mat(ctx, v_w, x);

    // T is the time dim of x; ggml stores [d_model, T] as ne = [d_model, T].
    const int T = (int)x->ne[1];

    Q = ggml_reshape_3d(ctx, Q, hd, n_q,  T);
    K = ggml_reshape_3d(ctx, K, hd, n_kv, T);
    V = ggml_reshape_3d(ctx, V, hd, n_kv, T);

    // NEOX RoPE on Q and K. Same args as the original inline code in
    // voxtral / voxtral4b / qwen3 / granite LLM blocks.
    Q = ggml_rope_ext(ctx, Q, positions, /*freq_factors*/nullptr,
                      hd, GGML_ROPE_TYPE_NEOX, n_ctx,
                      p.rope_theta, /*freq_scale*/1.0f, /*ext_factor*/0.0f,
                      /*attn_factor*/1.0f, /*beta_fast*/32.0f, /*beta_slow*/1.0f);
    K = ggml_rope_ext(ctx, K, positions, nullptr,
                      hd, GGML_ROPE_TYPE_NEOX, n_ctx,
                      p.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // GQA expansion: replicate each KV head `grp` times along a new dim so
    // K/V have n_heads rows instead of n_kv_heads, then flatten back.
    if (grp > 1) {
        ggml_tensor * K4 = ggml_reshape_4d(ctx, K, hd, 1, n_kv, T);
        ggml_tensor * V4 = ggml_reshape_4d(ctx, V, hd, 1, n_kv, T);
        K4 = ggml_repeat_4d(ctx, K4, hd, grp, n_kv, T);
        V4 = ggml_repeat_4d(ctx, V4, hd, grp, n_kv, T);
        K  = ggml_cont(ctx, ggml_reshape_3d(ctx, K4, hd, n_q, T));
        V  = ggml_cont(ctx, ggml_reshape_3d(ctx, V4, hd, n_q, T));
    }

    // Permute to flash-attention layout: (head_dim, T, n_heads).
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Flash attention. Output shape = (head_dim, n_heads, T, 1).
    ggml_tensor * attn = ggml_flash_attn_ext(
        ctx, Q, K, V, mask,
        p.attn_scale, /*max_bias*/0.0f, /*logit_softcap*/0.0f);

    // Back to (d_model, T).
    attn = ggml_reshape_2d(ctx, attn, hd * n_q, T);

    // Output projection (no bias).
    return ggml_mul_mat(ctx, o_w, attn);
}

} // namespace core_attn
