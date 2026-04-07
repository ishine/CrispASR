// cohere-main.cpp — CLI for Cohere Transcribe
//
// Usage matches whisper-cli conventions:
//   cohere-main -m MODEL.gguf -f audio.wav [-l en] [-t 4] [--verbose]
//
// By default only the transcript is written to stdout; all progress info
// goes to stderr and is suppressed unless --verbose is passed.

#include "cohere.h"
#include "common.h"
#include "common-whisper.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

struct cohere_params {
    std::string model;
    std::string fname_inp;
    std::string language       = "en";
    int         n_threads      = std::min(4, (int)std::thread::hardware_concurrency());
    int         verbosity      = 1;    // 0=silent 1=normal(loading only) 2=verbose(timing+steps)
    bool        use_flash      = false;
    bool        no_prints      = false;
    bool        debug          = false; // enables COHERE_DEBUG + COHERE_PROF env vars
    bool        no_punctuation = false; // use <|nopnc|> instead of <|pnc|>
    bool        output_txt     = false; // write transcript to <audio>.txt
};

static void print_usage(const char * prog) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] -m MODEL -f AUDIO\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,         --help           show this help message\n");
    fprintf(stderr, "  -m FNAME,   --model FNAME    path to cohere-transcribe.gguf\n");
    fprintf(stderr, "  -f FNAME,   --file FNAME     input audio file (WAV 16 kHz mono)\n");
    fprintf(stderr, "  -l LANG,    --language LANG  language code (default: en)\n");
    fprintf(stderr, "  -t N,       --threads N      number of threads (default: %d)\n",
            std::min(4, (int)std::thread::hardware_concurrency()));
    fprintf(stderr, "  -ot,        --output-txt     write transcript to <audio>.txt\n");
    fprintf(stderr, "  -npnc,      --no-punctuation disable punctuation in output\n");
    fprintf(stderr, "  -v,         --verbose        show timing info and per-step tokens\n");
    fprintf(stderr, "  -np,        --no-prints      suppress all informational output\n");
    fprintf(stderr, "  -d,         --debug          enable COHERE_DEBUG and COHERE_PROF\n");
    fprintf(stderr, "  --flash                      enable flash attention in decoder\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "supported languages:\n");
    fprintf(stderr, "  ar de el en es fr it ja ko nl pl pt vi zh\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "environment:\n");
    fprintf(stderr, "  COHERE_DEVICE=metal|cuda|cpu  force backend selection\n");
    fprintf(stderr, "  COHERE_THREADS=N              override thread count\n");
    fprintf(stderr, "  COHERE_DEBUG=1                verbose tensor/graph logging\n");
    fprintf(stderr, "  COHERE_PROF=1                 per-op profiling\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "input must be 16 kHz mono WAV; convert with:\n");
    fprintf(stderr, "  ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le audio.wav\n");
    fprintf(stderr, "\n");
}

static bool parse_args(int argc, char ** argv, cohere_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if ((arg == "-m" || arg == "--model")    && i+1 < argc) { p.model          = argv[++i];
        } else if ((arg == "-f" || arg == "--file")     && i+1 < argc) { p.fname_inp      = argv[++i];
        } else if ((arg == "-l" || arg == "--language") && i+1 < argc) { p.language       = argv[++i];
        } else if ((arg == "-t" || arg == "--threads")  && i+1 < argc) { p.n_threads      = std::atoi(argv[++i]);
        } else if (arg == "-ot"   || arg == "--output-txt")   { p.output_txt     = true;
        } else if (arg == "-npnc" || arg == "--no-punctuation"){ p.no_punctuation = true;
        } else if (arg == "-v"    || arg == "--verbose")       { p.verbosity      = 2;
        } else if (arg == "-np"   || arg == "--no-prints")     { p.no_prints      = true;
        } else if (arg == "-d"    || arg == "--debug")         { p.debug          = true;
        } else if (arg == "--flash")                           { p.use_flash      = true;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (p.model.empty() || p.fname_inp.empty()) {
        fprintf(stderr, "error: -m MODEL and -f AUDIO are required\n\n");
        print_usage(argv[0]);
        return false;
    }

    if (p.no_prints) p.verbosity = 0;

    return true;
}

// Derive output path: strip known audio extensions, append .txt
static std::string make_txt_path(const std::string & audio_path) {
    std::string base = audio_path;
    for (const char * ext : {".wav", ".WAV", ".mp3", ".MP3", ".flac", ".FLAC",
                              ".ogg", ".OGG", ".m4a", ".M4A", ".opus", ".OPUS"}) {
        if (base.size() > strlen(ext) &&
            base.compare(base.size() - strlen(ext), strlen(ext), ext) == 0) {
            base = base.substr(0, base.size() - strlen(ext));
            break;
        }
    }
    return base + ".txt";
}

int main(int argc, char ** argv) {
    cohere_params p;
    if (!parse_args(argc, argv, p)) return 1;

    // --debug: activate COHERE_DEBUG and COHERE_PROF environment variables.
    // setenv is POSIX; on Windows use _putenv_s.
    if (p.debug) {
#if defined(_WIN32)
        _putenv_s("COHERE_DEBUG", "1");
        _putenv_s("COHERE_PROF",  "1");
#else
        setenv("COHERE_DEBUG", "1", 1);
        setenv("COHERE_PROF",  "1", 1);
#endif
        p.verbosity = std::max(p.verbosity, 2);
    }

    // Build context params
    struct cohere_context_params params = cohere_context_default_params();
    params.n_threads       = p.n_threads;
    params.use_flash       = p.use_flash;
    params.no_punctuation  = p.no_punctuation;
    params.verbosity       = p.verbosity;

    if (p.verbosity >= 1) {
        fprintf(stderr, "%s: loading model '%s'\n", argv[0], p.model.c_str());
    }
    struct cohere_context * ctx = cohere_init_from_file(p.model.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to load model '%s'\n", argv[0], p.model.c_str());
        return 1;
    }

    // Load audio via common-whisper helper (handles stereo→mono downmix)
    std::vector<float> samples;
    std::vector<std::vector<float>> samples_stereo; // unused, mono only
    if (!read_audio_data(p.fname_inp, samples, samples_stereo, /*stereo=*/false)) {
        fprintf(stderr, "%s: failed to read audio '%s'\n", argv[0], p.fname_inp.c_str());
        cohere_free(ctx);
        return 1;
    }
    if (p.verbosity >= 1) {
        fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads\n",
                argv[0], p.fname_inp.c_str(),
                (int)samples.size(), (float)samples.size() / 16000.0f,
                p.n_threads);
    }

    // Transcribe
    char * text = cohere_transcribe(ctx, samples.data(), (int)samples.size(), p.language.c_str());

    if (!text) {
        fprintf(stderr, "%s: transcription failed\n", argv[0]);
        cohere_free(ctx);
        return 1;
    }

    // Write to stdout
    printf("%s\n", text);

    // Optionally write to .txt file
    if (p.output_txt) {
        const std::string txt_path = make_txt_path(p.fname_inp);
        std::ofstream ofs(txt_path);
        if (ofs) {
            ofs << text << "\n";
            if (p.verbosity >= 1) {
                fprintf(stderr, "%s: transcript written to '%s'\n", argv[0], txt_path.c_str());
            }
        } else {
            fprintf(stderr, "%s: warning: could not write to '%s'\n", argv[0], txt_path.c_str());
        }
    }

    free(text);

    if (p.verbosity >= 1) {
        fprintf(stderr, "\n%s: done\n", argv[0]);
    }

    cohere_free(ctx);
    return 0;
}
