// voxtral-test-llm — differential test for the Voxtral LLM forward.
//
// Loads:
//   - GGUF model from convert-voxtral-to-gguf.py
//   - voxtral_input_ids.npy   (T,)         int32   token IDs
//   - voxtral_logits.npy      (T, 131072)  f32     reference logits (last pos)
//
// Runs voxtral_run_llm() and reports max abs diff + per-position cosine
// similarity + top-1 match against the reference last-token logits.
//
// Usage:
//   voxtral-test-llm  voxtral-mini-3b-2507.gguf  /tmp/voxtral-ref

#include "voxtral.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

template <typename T>
static bool load_npy(const std::string & path, std::vector<T> & data,
                     std::vector<int> & shape, const char * dtype_str) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char magic[6]; f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint32_t hdr_len = 0;
    if (major == 1) { uint16_t hl; f.read((char*)&hl, 2); hdr_len = hl; }
    else            { f.read((char*)&hdr_len, 4); }
    std::string header(hdr_len, '\0'); f.read(&header[0], hdr_len);
    if (header.find(dtype_str) == std::string::npos) {
        fprintf(stderr, "%s: expected dtype %s\n", path.c_str(), dtype_str);
        return false;
    }
    auto sp = header.find("'shape':");
    auto lp = header.find('(', sp);
    auto rp = header.find(')', lp);
    std::string sh = header.substr(lp+1, rp-lp-1);
    shape.clear();
    size_t i = 0;
    while (i < sh.size()) {
        while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) i++;
        if (i >= sh.size()) break;
        int v = 0;
        while (i < sh.size() && sh[i] >= '0' && sh[i] <= '9') { v = v*10 + (sh[i]-'0'); i++; }
        shape.push_back(v);
    }
    size_t total = 1;
    for (int s : shape) total *= (size_t)s;
    data.resize(total);
    f.read((char*)data.data(), total * sizeof(T));
    return (bool)f;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s voxtral-mini-3b-2507.gguf REF_DIR\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    std::string  ref_dir    = argv[2];

    std::vector<int32_t> ids; std::vector<int> ids_shape;
    if (!load_npy<int32_t>(ref_dir + "/voxtral_input_ids.npy", ids, ids_shape, "'<i4'")) return 2;
    fprintf(stderr, "input_ids (%zu): ", ids.size());
    for (auto v : ids) fprintf(stderr, "%d ", v);
    fprintf(stderr, "\n");
    int T = (int)ids.size();

    std::vector<float> ref_logits; std::vector<int> ref_shape;
    if (!load_npy<float>(ref_dir + "/voxtral_logits.npy", ref_logits, ref_shape, "'<f4'")) return 3;
    fprintf(stderr, "ref logits shape: %d × %d\n", ref_shape[0], ref_shape[1]);
    int ref_T = ref_shape[0], ref_vocab = ref_shape[1];

    auto cp = voxtral_context_default_params();
    cp.n_threads = 4;
    auto * ctx = voxtral_init_from_file(model_path, cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 4; }

    int n_t = 0, vocab = 0;
    fprintf(stderr, "running LLM forward (T=%d) ...\n", T);
    float * logits = voxtral_run_llm(ctx, ids.data(), T, &n_t, &vocab);
    if (!logits) { fprintf(stderr, "run_llm failed\n"); voxtral_free(ctx); return 5; }
    fprintf(stderr, "C++ logits: %d × %d  (last-token-only)\n", n_t, vocab);

    if (vocab != ref_vocab) {
        fprintf(stderr, "vocab size mismatch: cpp=%d ref=%d\n", vocab, ref_vocab);
        free(logits); voxtral_free(ctx); return 6;
    }

    // The C++ side returns only the last token's logits; the reference dump
    // has all T positions. Compare against position T-1 of the reference.
    const float * ref_last = ref_logits.data() + (size_t)(ref_T - 1) * ref_vocab;

    // Argmax + cosine similarity
    int cpp_argmax = 0;
    int ref_argmax = 0;
    float cpp_max = -1e30f, ref_max = -1e30f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    float max_abs = 0.0f;
    int max_idx = -1;
    for (int k = 0; k < vocab; k++) {
        if (logits[k] > cpp_max) { cpp_max = logits[k]; cpp_argmax = k; }
        if (ref_last[k] > ref_max) { ref_max = ref_last[k]; ref_argmax = k; }
        double a = logits[k], b = ref_last[k];
        dot += a * b; na += a * a; nb += b * b;
        float ad = std::fabs(logits[k] - ref_last[k]);
        if (ad > max_abs) { max_abs = ad; max_idx = k; }
    }
    double cs = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    fprintf(stderr, "\nLAST-TOKEN LOGIT DIFF:\n");
    fprintf(stderr, "  cpp argmax:  %d  (logit %.4f)\n", cpp_argmax, cpp_max);
    fprintf(stderr, "  ref argmax:  %d  (logit %.4f)\n", ref_argmax, ref_max);
    fprintf(stderr, "  cosine sim:  %.6f\n", cs);
    fprintf(stderr, "  max abs:     %.4e (idx %d)\n", max_abs, max_idx);
    fprintf(stderr, "  match: %s\n", cpp_argmax == ref_argmax ? "✓" : "✗");

    // Top-5 from each side
    auto top5 = [&](const float * v) {
        std::vector<int> idx(vocab);
        for (int i = 0; i < vocab; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return v[a] > v[b]; });
        return idx;
    };
    auto cpp_top = top5(logits);
    auto ref_top = top5(ref_last);
    fprintf(stderr, "\n  cpp top-5: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%d ", cpp_top[i]);
    fprintf(stderr, "\n  ref top-5: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%d ", ref_top[i]);
    fprintf(stderr, "\n");

    int verdict = (cpp_argmax == ref_argmax && cs > 0.99) ? 0 : 1;
    fprintf(stderr, "\nverdict: %s\n", verdict == 0 ? "PASS" : "FAIL");
    free(logits);
    voxtral_free(ctx);
    return verdict;
}
