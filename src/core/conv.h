// src/core/conv.h — convolution helpers that work around ggml limitations.
//
// ggml has no `groups` argument on `ggml_conv_1d` or
// `ggml_conv_transpose_1d`, so any depthwise / grouped conv has to be
// open-coded. This header collects the specific shapes that come up
// repeatedly across the BigVGAN-family vocoder ports (Kokoro, future
// iSTFTNet variants, possibly mimo codec).
//
// Currently:
//   convt1d_depthwise_2x_k3  — depthwise ConvTranspose1d with kernel=3,
//                              stride=2, padding=1, output_padding=1.
//                              Used for 2× upsamples in iSTFTNet-style
//                              vocoder pool layers.

#pragma once

#include "ggml.h"

namespace core_convt {

// Depthwise ConvTranspose1d with parameters (k=3, s=2, p=1, op=1).
// Output length = 2 · T_in.
//
// PyTorch ConvTranspose1d emits `y[i] = sum input[j] · weight[k]` over
// (j, k) satisfying `j·stride + k − padding = i`. For our config:
//
//   y[c, 2t]   = w[c, 1] · x[c, t]                                  (j=t,   k=1)
//   y[c, 2t+1] = w[c, 2] · x[c, t] + w[c, 0] · x[c, t+1]            (j=t,k=2 + j=t+1,k=0)
//                                                                   (x[c, T]=0 boundary)
//
// **Critical**: `w[2]` and `w[0]` are NOT interchangeable in the odd
// case — getting the kernel ends swapped produces plausible-but-wrong
// audio that can survive informal QA. The Kokoro M11 diff harness
// caught exactly this bug (commit 448c1af); see LEARNINGS.md
// "Kokoro / StyleTTS2 lessons" Lesson 2.
//
// Inputs:
//   x        : (C, T)        F32, channel-major.
//   w_kernel : (K=3, 1, C)   F16, depthwise kernel (PyTorch
//              `nn.ConvTranspose1d(C, C, k=3, s=2, p=1, op=1, groups=C)`
//              stores weights as `(C, 1, K)` and the converter
//              transposes to `(K, 1, C)` for ggml).
//   w_bias   : (C,)          F32, optional per-channel bias (broadcast
//              over time). Pass nullptr to skip.
//
// Output: (C, 2·T) F32.
static inline ggml_tensor* convt1d_depthwise_2x_k3(ggml_context* ctx, ggml_tensor* x,
                                                   ggml_tensor* w_kernel, ggml_tensor* w_bias) {
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];

    // Permute kernel (K=3, 1, C) → (C, 3, 1), cast to F32 (F16 view + F32
    // mul fails on Metal at the kernel-dispatch level), reshape to
    // (C, 3), then take three column views w0/w1/w2.
    ggml_tensor* w_perm = ggml_cont(ctx, ggml_permute(ctx, w_kernel, 2, 0, 1, 3));   // (C, 3, 1) F16
    ggml_tensor* w_perm_f32 = ggml_cast(ctx, w_perm, GGML_TYPE_F32);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w_perm_f32, C, 3);                      // (C, 3) F32
    const size_t row_b = w_2d->nb[1];
    ggml_tensor* w0 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)0 * row_b);
    ggml_tensor* w1 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)1 * row_b);
    ggml_tensor* w2 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)2 * row_b);

    // x_shifted[c, t] = x[c, t+1] for t < T-1, 0 for t = T-1.
    // Take x[:, 1:] (C, T-1) and zero-pad on the right to (C, T).
    ggml_tensor* x_tail = ggml_view_2d(ctx, x, C, T - 1, x->nb[1], x->nb[1]);        // (C, T-1)
    x_tail = ggml_cont(ctx, x_tail);                                                  // contiguous
    ggml_tensor* x_shifted = ggml_pad_ext(ctx, x_tail, 0, 0, 0, 1, 0, 0, 0, 0);       // (C, T)

    // y_even (C, T) = w1 ⊙ x  (broadcast w1 over T)
    ggml_tensor* y_even = ggml_mul(ctx, x, w1);
    // y_odd (C, T) = w2 ⊙ x + w0 ⊙ x_shifted   (PyTorch ConvTranspose1d
    // kernel indexing — see derivation note above)
    ggml_tensor* y_odd = ggml_add(ctx, ggml_mul(ctx, x, w2), ggml_mul(ctx, x_shifted, w0));

    // Interleave: reshape both to (C, 1, T), concat dim=1 → (C, 2, T),
    // reshape to (C, 2T). Memory layout means consecutive time positions
    // alternate even/odd, which is the desired interleaving.
    ggml_tensor* even_3d = ggml_reshape_3d(ctx, y_even, C, 1, T);
    ggml_tensor* odd_3d  = ggml_reshape_3d(ctx, y_odd,  C, 1, T);
    ggml_tensor* stacked = ggml_concat(ctx, even_3d, odd_3d, /*dim=*/1);              // (C, 2, T)
    ggml_tensor* y = ggml_cont(ctx, ggml_reshape_2d(ctx, stacked, C, 2 * T));         // (C, 2T)

    if (w_bias)
        y = ggml_add(ctx, y, w_bias);
    return y;
}

} // namespace core_convt
