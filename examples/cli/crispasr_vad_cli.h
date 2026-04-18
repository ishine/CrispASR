// crispasr_vad_cli.h — CLI-side VAD shim.
//
// The VAD algorithmic core (Silero segmentation, merge/split, stitching,
// timestamp remapping) lives in `src/crispasr_vad.h` so every CrispASR
// consumer reaches it through the shared library. This CLI-local header
// re-exports those types by transitive include and adds one wrapper
// that translates CLI `whisper_params` (including auto-download policy)
// into a library call.
//
// CLI callers: use `crispasr_compute_audio_slices(... whisper_params &)`.
// Library callers / wrappers: use `crispasr_compute_vad_slices` +
// `crispasr_fixed_chunk_slices` from `src/crispasr_vad.h` directly.

#pragma once

#include "crispasr_vad.h" // from src/ via whisper target's PUBLIC include dir

#include <vector>

struct whisper_params; // fwd decl

// Build the list of audio slices for a CLI invocation.
//
// If `params.vad` or `params.vad_model` is set, resolves the VAD model
// path (auto-downloading the canonical Silero GGUF into the CLI cache
// when the user passed `--vad` without `--vad-model`), then calls the
// library's `crispasr_compute_vad_slices`. Otherwise falls back to
// `crispasr_fixed_chunk_slices(chunk_seconds)`.
//
// Returns an empty vector if VAD was requested but detected no speech.
std::vector<crispasr_audio_slice> crispasr_compute_audio_slices(const float* samples, int n_samples, int sample_rate,
                                                                int chunk_seconds, const whisper_params& params);
