// canary-main — minimal CLI for nvidia/canary-1b-v2.
//
// Iteration 1: loader-only smoke test. Loads the GGUF, reports hparams,
// confirms the special tokens needed for the decoder prompt resolve.

#include "canary.h"
#include "common-whisper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\nusage: %s -m MODEL.gguf [-f AUDIO.wav] [-sl LANG] [-tl LANG]\n\n"
        "options:\n"
        "  -h, --help        show this help\n"
        "  -m FNAME          canary GGUF model (from convert-canary-to-gguf.py)\n"
        "  -f FNAME          input audio (16 kHz mono WAV)  [stub for now]\n"
        "  -sl LANG          source language ISO-639-1 (e.g. en, de, fr)\n"
        "  -tl LANG          target language: same as -sl → ASR; differs → translation\n"
        "  -t  N             threads (default: 4)\n\n",
        prog);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string audio_path;
    std::string source_lang = "en";
    std::string target_lang = "en";
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "-m"  && i+1 < argc) model_path  = argv[++i];
        else if (a == "-f"  && i+1 < argc) audio_path  = argv[++i];
        else if (a == "-sl" && i+1 < argc) source_lang = argv[++i];
        else if (a == "-tl" && i+1 < argc) target_lang = argv[++i];
        else if (a == "-t"  && i+1 < argc) n_threads   = std::atoi(argv[++i]);
        else if (a == "-v"  || a == "--verbose") {/* set below */}
        else { fprintf(stderr, "unknown option '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (model_path.empty()) { print_usage(argv[0]); return 1; }

    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose") verbose = true;
    }

    canary_context_params p = canary_context_default_params();
    p.n_threads = n_threads;
    p.verbosity = verbose ? 2 : 1;

    canary_context * ctx = canary_init_from_file(model_path.c_str(), p);
    if (!ctx) {
        fprintf(stderr, "%s: failed to load '%s'\n", argv[0], model_path.c_str());
        return 1;
    }

    canary_test_load(ctx);

    fprintf(stderr, "\n%s: encoder smoke test ...\n", argv[0]);
    if (canary_test_encoder(ctx, 100) < 0) {
        fprintf(stderr, "%s: encoder smoke test FAILED\n", argv[0]);
        canary_free(ctx);
        return 2;
    }

    fprintf(stderr, "\n%s: would transcribe with src='%s' tgt='%s' (mode: %s)\n",
            argv[0], source_lang.c_str(), target_lang.c_str(),
            source_lang == target_lang ? "ASR" : "speech translation");

    if (!audio_path.empty()) {
        std::vector<float> pcm;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(audio_path, pcm, stereo, /*stereo=*/false)) {
            fprintf(stderr, "%s: failed to read '%s'\n", argv[0], audio_path.c_str());
            canary_free(ctx);
            return 3;
        }
        fprintf(stderr, "%s: audio loaded — %d samples (%.2fs) @ %d Hz\n",
                argv[0], (int)pcm.size(),
                (double)pcm.size() / canary_sample_rate(ctx),
                canary_sample_rate(ctx));

        canary_result * r = canary_transcribe_ex(ctx, pcm.data(), (int)pcm.size(),
                                                 source_lang.c_str(), target_lang.c_str(),
                                                 /*punctuation=*/true, /*t_offset_cs=*/0);
        if (!r) {
            fprintf(stderr, "%s: transcription failed\n", argv[0]);
            canary_free(ctx);
            return 4;
        }
        printf("%s\n", r->text ? r->text : "");
        canary_result_free(r);
    }

    canary_free(ctx);
    return 0;
}
