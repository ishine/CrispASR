// audioseal.cpp — AudioSeal watermark generator & detector (ggml implementation).
//
// SEANet architecture (encoder-decoder with residual blocks, ELU activations,
// and LSTM). The generator embeds a watermark; the detector recovers it.
//
// Key differences from SNAC/EnCodec codecs:
//   - No quantizer/codebook — this is a continuous autoencoder
//   - Bidirectional LSTM between encoder and decoder
//   - Message embedding via learned linear projection added at bottleneck
//   - ELU activations instead of Snake
//   - Additive watermark: output = input + generator_output
//
// Tensor layout: (C, T) channels-innermost, matching the SNAC convention.
// Conv ops transpose to (T, C) for ggml and back.

#include "audioseal.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct audioseal_hparams {
    uint32_t sample_rate = 16000;
    uint32_t channels = 1;        // mono
    uint32_t dimension = 128;     // encoder/decoder base dim (SEANet default)
    uint32_t n_filters = 32;     // first layer channels
    uint32_t n_residual_layers = 1;
    uint32_t ratios_n = 4;       // number of encoder/decoder blocks
    uint32_t lstm_layers = 2;
    uint32_t nbits = 16;         // watermark message bits
    uint32_t hop_length = 1;     // computed from ratios product
    std::vector<uint32_t> ratios; // downsampling ratios [8, 5, 4, 2] → hop=320
};

// ---------------------------------------------------------------------------
// Layer weight structs
// ---------------------------------------------------------------------------

struct audioseal_res_unit {
    // ELU → Conv1d(C,C,k=3,dil) → ELU → Conv1d(C,C,k=1) + shortcut
    ggml_tensor* conv0_w = nullptr; // (3, C_in, C_out) dilation=d
    ggml_tensor* conv0_b = nullptr;
    ggml_tensor* conv1_w = nullptr; // (1, C_in, C_out) pointwise
    ggml_tensor* conv1_b = nullptr;
    // Optional shortcut 1×1 conv (when dims differ)
    ggml_tensor* short_w = nullptr;
    ggml_tensor* short_b = nullptr;
};

struct audioseal_enc_block {
    std::vector<audioseal_res_unit> res; // typically 1 residual layer
    ggml_tensor* down_w = nullptr;       // Conv1d downsample (k=2*ratio, s=ratio)
    ggml_tensor* down_b = nullptr;
};

struct audioseal_dec_block {
    ggml_tensor* up_w = nullptr;         // ConvTranspose1d upsample
    ggml_tensor* up_b = nullptr;
    std::vector<audioseal_res_unit> res;
};

struct audioseal_lstm_layer {
    // Weights for input gate, forget gate, cell gate, output gate
    // Combined as (4*hidden, input) for weight_ih and (4*hidden, hidden) for weight_hh
    ggml_tensor* weight_ih = nullptr;
    ggml_tensor* bias_ih = nullptr;
    ggml_tensor* weight_hh = nullptr;
    ggml_tensor* bias_hh = nullptr;
};

// ---------------------------------------------------------------------------
// Graph building helpers
// ---------------------------------------------------------------------------

// ELU activation: y = x if x >= 0, else alpha*(exp(x)-1). Alpha=1.0.
static ggml_tensor* elu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_elu(ctx, x);
}

// Conv1d wrapper: (C_in, T) → (C_out, T_out).
// w shape: (K, C_in, C_out) in GGUF (ggml conv1d convention).
static ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* x,
                           ggml_tensor* w, ggml_tensor* b,
                           int stride, int padding, int dilation) {
    // ggml_conv_1d expects x=(T, C_in), w=(K, C_in, C_out)
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C_in)
    ggml_tensor* y = ggml_conv_1d(ctx, w, xt, stride, padding, dilation);
    // y is (T_out, C_out), transpose back to (C_out, T_out)
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, (int)b->ne[0], 1));
    }
    return y;
}

// LSTM forward pass for one layer. Input x: (D, T), returns (D, T).
// Bidirectional is handled by running forward + backward and summing.
static ggml_tensor* lstm_forward(ggml_context* ctx, ggml_tensor* x,
                                 audioseal_lstm_layer& layer, int hidden_dim) {
    // For the initial implementation, we use a simplified LSTM that
    // processes the sequence. Full LSTM with gates would require a
    // custom ggml op or loop unrolling. For now, approximate with
    // a linear projection (the LSTM weights are still loaded and
    // available for a proper implementation).
    //
    // TODO: implement proper LSTM gate computation when ggml adds
    // native LSTM support, or unroll for short sequences.

    // Simplified: treat as a linear projection (captures the learned
    // transformation without temporal gating)
    // weight_ih: (4*D, D) → take first D rows as projection
    const int D = hidden_dim;
    ggml_tensor* w = ggml_view_2d(ctx, layer.weight_ih,
                                   D, D, layer.weight_ih->nb[1], 0);
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    if (layer.bias_ih) {
        ggml_tensor* b = ggml_view_1d(ctx, layer.bias_ih, D, 0);
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, D, 1));
    }
    return ggml_tanh(ctx, y);
}

} // namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct audioseal_ctx {
    audioseal_params params{};
    audioseal_hparams hp;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Generator encoder
    ggml_tensor* gen_enc_in_w = nullptr; // Conv1d(1, n_filters, k=7)
    ggml_tensor* gen_enc_in_b = nullptr;
    std::vector<audioseal_enc_block> gen_enc_blocks;
    std::vector<audioseal_lstm_layer> gen_lstm;
    ggml_tensor* gen_enc_out_w = nullptr; // Conv1d(C, D, k=7)
    ggml_tensor* gen_enc_out_b = nullptr;

    // Generator message projection
    ggml_tensor* gen_msg_w = nullptr; // Linear(nbits, D)
    ggml_tensor* gen_msg_b = nullptr;

    // Generator decoder
    ggml_tensor* gen_dec_in_w = nullptr;  // Conv1d(D, C, k=7)
    ggml_tensor* gen_dec_in_b = nullptr;
    std::vector<audioseal_dec_block> gen_dec_blocks;
    ggml_tensor* gen_dec_out_w = nullptr; // Conv1d(n_filters, 1, k=7)
    ggml_tensor* gen_dec_out_b = nullptr;

    // Detector encoder (same architecture as generator encoder)
    ggml_tensor* det_enc_in_w = nullptr;
    ggml_tensor* det_enc_in_b = nullptr;
    std::vector<audioseal_enc_block> det_enc_blocks;
    std::vector<audioseal_lstm_layer> det_lstm;

    // Detector heads
    ggml_tensor* det_linear_w = nullptr; // Linear(D, 2) — detection
    ggml_tensor* det_linear_b = nullptr;
    ggml_tensor* det_msg_w = nullptr;    // Linear(D, nbits) — message
    ggml_tensor* det_msg_b = nullptr;

    bool has_generator = false;
    bool has_detector = false;

    std::vector<uint8_t> compute_meta;

    ~audioseal_ctx() {
        if (ctx_w)
            ggml_free(ctx_w);
        if (buf_w)
            ggml_backend_buffer_free(buf_w);
        if (backend && backend != backend_cpu)
            ggml_backend_free(backend);
        if (backend_cpu)
            ggml_backend_free(backend_cpu);
    }
};

// ---------------------------------------------------------------------------
// Metadata + tensor loading
// ---------------------------------------------------------------------------

namespace {

static void load_metadata(audioseal_ctx* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.sample_rate = core_gguf::kv_u32(g, "audioseal.sample_rate", hp.sample_rate);
    hp.dimension = core_gguf::kv_u32(g, "audioseal.dimension", hp.dimension);
    hp.n_filters = core_gguf::kv_u32(g, "audioseal.n_filters", hp.n_filters);
    hp.n_residual_layers = core_gguf::kv_u32(g, "audioseal.n_residual_layers", hp.n_residual_layers);
    hp.nbits = core_gguf::kv_u32(g, "audioseal.nbits", hp.nbits);
    hp.lstm_layers = core_gguf::kv_u32(g, "audioseal.lstm_layers", hp.lstm_layers);

    // Read ratios array
    const int k = gguf_find_key(g, "audioseal.ratios");
    if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_ARRAY) {
        const int n = gguf_get_arr_n(g, k);
        hp.ratios.resize((size_t)n);
        const auto* d = (const uint32_t*)gguf_get_arr_data(g, k);
        hp.hop_length = 1;
        for (int i = 0; i < n; i++) {
            hp.ratios[i] = d[i];
            hp.hop_length *= d[i];
        }
        hp.ratios_n = (uint32_t)n;
    } else {
        // Default AudioSeal ratios
        hp.ratios = {8, 5, 4, 2};
        hp.ratios_n = 4;
        hp.hop_length = 320;
    }
}

static bool bind_enc_block(std::map<std::string, ggml_tensor*>& t,
                           const std::string& prefix,
                           audioseal_enc_block& blk,
                           int n_res, const char* tag) {
    for (int r = 0; r < n_res; r++) {
        audioseal_res_unit ru;
        std::string rp = prefix + ".res." + std::to_string(r);
        ru.conv0_w = core_gguf::try_get(t, (rp + ".conv0.weight").c_str());
        ru.conv0_b = core_gguf::try_get(t, (rp + ".conv0.bias").c_str());
        ru.conv1_w = core_gguf::try_get(t, (rp + ".conv1.weight").c_str());
        ru.conv1_b = core_gguf::try_get(t, (rp + ".conv1.bias").c_str());
        ru.short_w = core_gguf::try_get(t, (rp + ".shortcut.weight").c_str());
        ru.short_b = core_gguf::try_get(t, (rp + ".shortcut.bias").c_str());
        blk.res.push_back(ru);
    }
    blk.down_w = core_gguf::try_get(t, (prefix + ".down.weight").c_str());
    blk.down_b = core_gguf::try_get(t, (prefix + ".down.bias").c_str());
    return true;
}

static bool bind_dec_block(std::map<std::string, ggml_tensor*>& t,
                           const std::string& prefix,
                           audioseal_dec_block& blk,
                           int n_res, const char* tag) {
    blk.up_w = core_gguf::try_get(t, (prefix + ".up.weight").c_str());
    blk.up_b = core_gguf::try_get(t, (prefix + ".up.bias").c_str());
    for (int r = 0; r < n_res; r++) {
        audioseal_res_unit ru;
        std::string rp = prefix + ".res." + std::to_string(r);
        ru.conv0_w = core_gguf::try_get(t, (rp + ".conv0.weight").c_str());
        ru.conv0_b = core_gguf::try_get(t, (rp + ".conv0.bias").c_str());
        ru.conv1_w = core_gguf::try_get(t, (rp + ".conv1.weight").c_str());
        ru.conv1_b = core_gguf::try_get(t, (rp + ".conv1.bias").c_str());
        ru.short_w = core_gguf::try_get(t, (rp + ".shortcut.weight").c_str());
        ru.short_b = core_gguf::try_get(t, (rp + ".shortcut.bias").c_str());
        blk.res.push_back(ru);
    }
    return true;
}

static bool bind_lstm(std::map<std::string, ggml_tensor*>& t,
                      const std::string& prefix,
                      std::vector<audioseal_lstm_layer>& layers,
                      int n_layers) {
    layers.resize((size_t)n_layers);
    for (int i = 0; i < n_layers; i++) {
        std::string lp = prefix + "." + std::to_string(i);
        layers[i].weight_ih = core_gguf::try_get(t, (lp + ".weight_ih").c_str());
        layers[i].bias_ih = core_gguf::try_get(t, (lp + ".bias_ih").c_str());
        layers[i].weight_hh = core_gguf::try_get(t, (lp + ".weight_hh").c_str());
        layers[i].bias_hh = core_gguf::try_get(t, (lp + ".bias_hh").c_str());
    }
    return true;
}

static bool bind_tensors(audioseal_ctx* c) {
    auto& t = c->tensors;
    const char* tag = "audioseal";
    const int n_res = (int)c->hp.n_residual_layers;
    const int n_blocks = (int)c->hp.ratios_n;

    // --- Generator ---
    c->gen_enc_in_w = core_gguf::try_get(t, "audioseal.gen.enc.in.weight");
    c->gen_enc_in_b = core_gguf::try_get(t, "audioseal.gen.enc.in.bias");
    if (c->gen_enc_in_w) {
        c->has_generator = true;

        c->gen_enc_blocks.resize((size_t)n_blocks);
        for (int i = 0; i < n_blocks; i++) {
            std::string prefix = "audioseal.gen.enc.blk." + std::to_string(i);
            bind_enc_block(t, prefix, c->gen_enc_blocks[i], n_res, tag);
        }
        bind_lstm(t, "audioseal.gen.lstm", c->gen_lstm, (int)c->hp.lstm_layers);

        c->gen_enc_out_w = core_gguf::try_get(t, "audioseal.gen.enc.out.weight");
        c->gen_enc_out_b = core_gguf::try_get(t, "audioseal.gen.enc.out.bias");

        c->gen_msg_w = core_gguf::try_get(t, "audioseal.gen.msg.weight");
        c->gen_msg_b = core_gguf::try_get(t, "audioseal.gen.msg.bias");

        c->gen_dec_in_w = core_gguf::try_get(t, "audioseal.gen.dec.in.weight");
        c->gen_dec_in_b = core_gguf::try_get(t, "audioseal.gen.dec.in.bias");
        c->gen_dec_blocks.resize((size_t)n_blocks);
        for (int i = 0; i < n_blocks; i++) {
            std::string prefix = "audioseal.gen.dec.blk." + std::to_string(i);
            bind_dec_block(t, prefix, c->gen_dec_blocks[i], n_res, tag);
        }
        c->gen_dec_out_w = core_gguf::try_get(t, "audioseal.gen.dec.out.weight");
        c->gen_dec_out_b = core_gguf::try_get(t, "audioseal.gen.dec.out.bias");
    }

    // --- Detector ---
    c->det_enc_in_w = core_gguf::try_get(t, "audioseal.det.enc.in.weight");
    c->det_enc_in_b = core_gguf::try_get(t, "audioseal.det.enc.in.bias");
    if (c->det_enc_in_w) {
        c->has_detector = true;

        c->det_enc_blocks.resize((size_t)n_blocks);
        for (int i = 0; i < n_blocks; i++) {
            std::string prefix = "audioseal.det.enc.blk." + std::to_string(i);
            bind_enc_block(t, prefix, c->det_enc_blocks[i], n_res, tag);
        }
        bind_lstm(t, "audioseal.det.lstm", c->det_lstm, (int)c->hp.lstm_layers);

        c->det_linear_w = core_gguf::try_get(t, "audioseal.det.linear.weight");
        c->det_linear_b = core_gguf::try_get(t, "audioseal.det.linear.bias");
        c->det_msg_w = core_gguf::try_get(t, "audioseal.det.msg.weight");
        c->det_msg_b = core_gguf::try_get(t, "audioseal.det.msg.bias");
    }

    if (!c->has_generator && !c->has_detector) {
        fprintf(stderr, "%s: no generator or detector tensors found in GGUF\n", tag);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Graph building: residual unit
// ---------------------------------------------------------------------------

static ggml_tensor* build_res_unit(ggml_context* ctx, ggml_tensor* x,
                                   const audioseal_res_unit& ru, int dilation) {
    ggml_tensor* y = elu(ctx, x);
    if (ru.conv0_w) {
        int pad = dilation; // k=3, dilation=d → pad=d for same-size
        y = conv1d(ctx, y, ru.conv0_w, ru.conv0_b, 1, pad, dilation);
    }
    y = elu(ctx, y);
    if (ru.conv1_w) {
        y = conv1d(ctx, y, ru.conv1_w, ru.conv1_b, 1, 0, 1);
    }
    // Shortcut
    ggml_tensor* skip = x;
    if (ru.short_w) {
        skip = conv1d(ctx, x, ru.short_w, ru.short_b, 1, 0, 1);
    }
    return ggml_add(ctx, y, skip);
}

// ---------------------------------------------------------------------------
// Graph building: encoder
// ---------------------------------------------------------------------------

static ggml_tensor* build_encoder(ggml_context* ctx, audioseal_ctx* c,
                                  ggml_tensor* x,
                                  ggml_tensor* enc_in_w, ggml_tensor* enc_in_b,
                                  std::vector<audioseal_enc_block>& blocks,
                                  std::vector<audioseal_lstm_layer>& lstm,
                                  ggml_tensor* enc_out_w, ggml_tensor* enc_out_b) {
    // Input conv: Conv1d(1, n_filters, k=7, p=3)
    if (enc_in_w) {
        x = conv1d(ctx, x, enc_in_w, enc_in_b, 1, 3, 1);
    }

    // Encoder blocks
    static const int dilations[] = {1, 3, 9};
    for (size_t i = 0; i < blocks.size(); i++) {
        auto& blk = blocks[i];
        // Residual units
        for (size_t r = 0; r < blk.res.size(); r++) {
            int dil = (r < 3) ? dilations[r] : 1;
            x = build_res_unit(ctx, x, blk.res[r], dil);
        }
        // Downsample
        if (blk.down_w) {
            int ratio = (int)c->hp.ratios[i];
            int pad = ratio / 2; // symmetric padding
            x = conv1d(ctx, x, blk.down_w, blk.down_b, ratio, pad, 1);
        }
    }

    // LSTM
    int D = (int)x->ne[0];
    for (auto& layer : lstm) {
        if (layer.weight_ih)
            x = lstm_forward(ctx, x, layer, D);
    }

    // Output conv
    if (enc_out_w) {
        x = conv1d(ctx, x, enc_out_w, enc_out_b, 1, 3, 1);
    }

    return x;
}

// ---------------------------------------------------------------------------
// Graph building: decoder
// ---------------------------------------------------------------------------

static ggml_tensor* build_decoder(ggml_context* ctx, audioseal_ctx* c,
                                  ggml_tensor* x,
                                  ggml_tensor* dec_in_w, ggml_tensor* dec_in_b,
                                  std::vector<audioseal_dec_block>& blocks,
                                  ggml_tensor* dec_out_w, ggml_tensor* dec_out_b) {
    // Input conv
    if (dec_in_w) {
        x = conv1d(ctx, x, dec_in_w, dec_in_b, 1, 3, 1);
    }

    // Decoder blocks (reversed ratios for upsampling)
    static const int dilations[] = {1, 3, 9};
    for (size_t i = 0; i < blocks.size(); i++) {
        auto& blk = blocks[i];
        // Upsample via ConvTranspose1d
        if (blk.up_w) {
            int ratio = (int)c->hp.ratios[c->hp.ratios_n - 1 - i]; // reversed
            ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
            ggml_tensor* y = ggml_conv_transpose_1d(ctx, blk.up_w, xt, ratio, 0, 1);
            y = ggml_cont(ctx, ggml_transpose(ctx, y));
            // Crop to expected size
            int expected_t = (int)x->ne[1] * ratio;
            if ((int)y->ne[1] > expected_t) {
                int crop = ((int)y->ne[1] - expected_t) / 2;
                y = ggml_view_2d(ctx, y, (int)y->ne[0], expected_t,
                                 y->nb[1], crop * y->nb[1]);
                y = ggml_cont(ctx, y);
            }
            if (blk.up_b) {
                y = ggml_add(ctx, y, ggml_reshape_2d(ctx, blk.up_b, (int)blk.up_b->ne[0], 1));
            }
            x = y;
        }
        // Residual units
        for (size_t r = 0; r < blk.res.size(); r++) {
            int dil = (r < 3) ? dilations[r] : 1;
            x = build_res_unit(ctx, x, blk.res[r], dil);
        }
    }

    // Output conv
    if (dec_out_w) {
        x = conv1d(ctx, x, dec_out_w, dec_out_b, 1, 3, 1);
    }

    // Tanh output
    x = ggml_tanh(ctx, x);
    return x;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

struct audioseal_params audioseal_default_params(void) {
    return {/*.n_threads=*/4, /*.verbosity=*/1, /*.use_gpu=*/false};
}

struct audioseal_ctx* audioseal_init_from_file(const char* path, struct audioseal_params params) {
    auto* c = new audioseal_ctx;
    c->params = params;

    // Pass 1: metadata
    gguf_context* g = core_gguf::open_metadata(path);
    if (!g) {
        fprintf(stderr, "audioseal: cannot open '%s'\n", path);
        delete c;
        return nullptr;
    }
    load_metadata(c, g);
    core_gguf::free_metadata(g);

    // Backend
    if (params.use_gpu) {
        c->backend = ggml_backend_init_best();
    }
    if (!c->backend) {
        c->backend = ggml_backend_cpu_init();
    }
    c->backend_cpu = ggml_backend_cpu_init();

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "audioseal", wl)) {
        fprintf(stderr, "audioseal: weight loading failed\n");
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_tensors(c)) {
        delete c;
        return nullptr;
    }

    // Allocate compute scratch (generous for ~5M param model)
    c->compute_meta.resize(256 * 1024 * 1024); // 256 MB

    if (params.verbosity > 0) {
        fprintf(stderr, "audioseal: loaded from '%s' — generator=%s detector=%s "
                        "sr=%u nbits=%u hop=%u ratios=[",
                path, c->has_generator ? "yes" : "no",
                c->has_detector ? "yes" : "no",
                c->hp.sample_rate, c->hp.nbits, c->hp.hop_length);
        for (size_t i = 0; i < c->hp.ratios.size(); i++) {
            if (i > 0) fprintf(stderr, ",");
            fprintf(stderr, "%u", c->hp.ratios[i]);
        }
        fprintf(stderr, "] tensors=%zu\n", c->tensors.size());
    }
    return c;
}

void audioseal_free(struct audioseal_ctx* ctx) {
    delete ctx;
}

uint32_t audioseal_sample_rate(const struct audioseal_ctx* ctx) {
    return ctx ? ctx->hp.sample_rate : 16000;
}

uint32_t audioseal_nbits(const struct audioseal_ctx* ctx) {
    return ctx ? ctx->hp.nbits : 16;
}

float* audioseal_embed(struct audioseal_ctx* ctx,
                       const float* pcm, int n_samples,
                       const uint8_t* message) {
    if (!ctx || !pcm || n_samples <= 0 || !ctx->has_generator)
        return nullptr;

    // Build compute graph
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) return nullptr;

    ggml_cgraph* gf = ggml_new_graph(ctx0);

    // Input tensor: (1, T) mono audio
    ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_samples);
    ggml_set_name(x_in, "audio_in");
    ggml_set_input(x_in);

    // Message tensor: (nbits,)
    ggml_tensor* msg = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int)ctx->hp.nbits);
    ggml_set_name(msg, "message_in");
    ggml_set_input(msg);

    // Encoder
    ggml_tensor* latent = build_encoder(ctx0, ctx, x_in,
                                         ctx->gen_enc_in_w, ctx->gen_enc_in_b,
                                         ctx->gen_enc_blocks, ctx->gen_lstm,
                                         ctx->gen_enc_out_w, ctx->gen_enc_out_b);

    // Message projection: expand message to match latent time dimension
    if (ctx->gen_msg_w) {
        ggml_tensor* msg_proj = ggml_mul_mat(ctx0, ctx->gen_msg_w, msg);
        if (ctx->gen_msg_b) {
            msg_proj = ggml_add(ctx0, msg_proj, ctx->gen_msg_b);
        }
        // Reshape to (D, 1) and broadcast-add to latent (D, T_latent)
        int D = (int)latent->ne[0];
        msg_proj = ggml_reshape_2d(ctx0, msg_proj, D, 1);
        latent = ggml_add(ctx0, latent, msg_proj);
    }

    // Decoder
    ggml_tensor* wm = build_decoder(ctx0, ctx, latent,
                                     ctx->gen_dec_in_w, ctx->gen_dec_in_b,
                                     ctx->gen_dec_blocks,
                                     ctx->gen_dec_out_w, ctx->gen_dec_out_b);

    // Output = input + watermark (additive)
    ggml_tensor* output = ggml_add(ctx0, x_in, wm);
    ggml_set_name(output, "audio_out");
    ggml_set_output(output);
    ggml_build_forward_expand(gf, output);

    // Allocate + compute
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "audioseal: graph allocation failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Set inputs
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "audio_in"),
                            pcm, 0, (size_t)n_samples * sizeof(float));

    // Set message (default: all ones)
    std::vector<float> msg_vec(ctx->hp.nbits, 1.0f);
    if (message) {
        for (uint32_t i = 0; i < ctx->hp.nbits; i++)
            msg_vec[i] = message[i] ? 1.0f : 0.0f;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "message_in"),
                            msg_vec.data(), 0, ctx->hp.nbits * sizeof(float));

    ggml_status st = ggml_backend_graph_compute(ctx->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "audioseal: graph compute failed (status %d)\n", (int)st);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Extract output
    ggml_tensor* out = ggml_graph_get_tensor(gf, "audio_out");
    float* result = (float*)std::malloc((size_t)n_samples * sizeof(float));
    if (result) {
        ggml_backend_tensor_get(out, result, 0, (size_t)n_samples * sizeof(float));
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}

float* audioseal_detect(struct audioseal_ctx* ctx,
                        const float* pcm, int n_samples,
                        int* out_n, uint8_t* out_message) {
    if (!ctx || !pcm || n_samples <= 0 || !ctx->has_detector)
        return nullptr;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) return nullptr;

    ggml_cgraph* gf = ggml_new_graph(ctx0);

    // Input
    ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_samples);
    ggml_set_name(x_in, "audio_in");
    ggml_set_input(x_in);

    // Detector encoder
    ggml_tensor* latent = build_encoder(ctx0, ctx, x_in,
                                         ctx->det_enc_in_w, ctx->det_enc_in_b,
                                         ctx->det_enc_blocks, ctx->det_lstm,
                                         nullptr, nullptr);

    // Detection head: Linear(D, 2) → softmax → take watermark probability
    ggml_tensor* det_logits = latent;
    if (ctx->det_linear_w) {
        det_logits = ggml_mul_mat(ctx0, ctx->det_linear_w, latent);
        if (ctx->det_linear_b) {
            det_logits = ggml_add(ctx0, det_logits,
                                  ggml_reshape_2d(ctx0, ctx->det_linear_b, 2, 1));
        }
    }
    // Softmax along dim=0 (2 classes), take index 1 (watermark present)
    det_logits = ggml_soft_max(ctx0, det_logits);
    ggml_tensor* det_probs = ggml_view_2d(ctx0, det_logits, 1, (int)det_logits->ne[1],
                                           det_logits->nb[1], sizeof(float));
    det_probs = ggml_cont(ctx0, det_probs);
    ggml_set_name(det_probs, "det_probs");
    ggml_set_output(det_probs);
    ggml_build_forward_expand(gf, det_probs);

    // Message head (optional)
    ggml_tensor* msg_out = nullptr;
    if (ctx->det_msg_w && out_message) {
        // Average latent over time → (D, 1) → Linear → sigmoid
        ggml_tensor* latent_mean = ggml_pool_1d(ctx0, latent, GGML_OP_POOL_AVG,
                                                  (int)latent->ne[1], (int)latent->ne[1], 0);
        msg_out = ggml_mul_mat(ctx0, ctx->det_msg_w, latent_mean);
        if (ctx->det_msg_b) {
            msg_out = ggml_add(ctx0, msg_out, ctx->det_msg_b);
        }
        msg_out = ggml_sigmoid(ctx0, msg_out);
        ggml_set_name(msg_out, "msg_out");
        ggml_set_output(msg_out);
        ggml_build_forward_expand(gf, msg_out);
    }

    // Allocate + compute
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "audio_in"),
                            pcm, 0, (size_t)n_samples * sizeof(float));

    ggml_status st = ggml_backend_graph_compute(ctx->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Extract detection probabilities
    ggml_tensor* probs = ggml_graph_get_tensor(gf, "det_probs");
    int n_frames = (int)probs->ne[1];
    float* result = (float*)std::malloc((size_t)n_frames * sizeof(float));
    if (result) {
        ggml_backend_tensor_get(probs, result, 0, (size_t)n_frames * sizeof(float));
    }
    if (out_n) *out_n = n_frames;

    // Extract message bits
    if (msg_out && out_message) {
        ggml_tensor* mo = ggml_graph_get_tensor(gf, "msg_out");
        if (mo) {
            std::vector<float> msg_probs(ctx->hp.nbits);
            ggml_backend_tensor_get(mo, msg_probs.data(), 0, ctx->hp.nbits * sizeof(float));
            for (uint32_t i = 0; i < ctx->hp.nbits; i++) {
                out_message[i] = msg_probs[i] > 0.5f ? 1 : 0;
            }
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}
