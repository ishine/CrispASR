# Voxtral implementation comparison: CrispASR vs max-lt/voxtral-cpp vs llama.cpp mtmd

Three independent C++ implementations of Voxtral-Mini-3B inference,
compared across architecture, correctness, performance, and ecosystem.

## The three approaches

### 1. CrispASR (this repo) — standalone ggml runtime

- **Architecture:** Self-contained encoder + projector + LLM forward,
  all written directly against the ggml API. Single GGUF file bundles
  model weights + mel filterbank + Tekken vocab blob + all metadata.
- **Code:** ~1300 LOC in `src/voxtral.cpp` — conv1d front-end, 32-layer
  Whisper encoder, 4-frame-stack projector, 30-layer Llama 3 LLM, F16
  KV cache, flash-attn on both prefill and decode paths.
- **Correctness:** Diff-tested against PyTorch reference at every stage
  (LLM top-5 5/5 match, encoder cosine >0.99, end-to-end JFK transcript
  correct). No known correctness issues.
- **Performance (Q4_K, 4 threads, jfk 11s):** 70 s total, 242 ms/token.

### 2. max-lt/voxtral-cpp — thin wrapper around llama.cpp + mtmd

- **Architecture:** 741-LOC CLI that `FetchContent`s llama.cpp at build
  time and delegates ALL inference to llama.cpp's `mtmd` (multimodal
  toolkit) API. The actual transcribe function is ~100 LOC: `mtmd_bitmap
  → mtmd_tokenize → mtmd_helper_eval_chunks → llama_sampler`.
- **GGUF:** Two files — main model GGUF (from llama.cpp's converter) +
  separate mmproj GGUF for the audio projector. Standard llama.cpp format.
- **Features:** Interactive terminal model selector, microphone recording
  via miniaudio, audio enhancement (DC removal, highpass, noise gate,
  normalization), JSON-prefill transcription prompt, support for both
  the 3B and 24B models.
- **Bug workaround:** Ships a patch for llama.cpp's mtmd.cpp to add the
  `[BEGIN_AUDIO]` token for Voxtral (upstream issue
  [#17868](https://github.com/ggml-org/llama.cpp/issues/17868), ignored
  by maintainers as of writing).

### 3. steampunque / native llama.cpp mtmd — standard llama.cpp tooling

- **Architecture:** Uses llama.cpp's built-in `convert_hf_to_gguf.py` +
  the `mtmd` multimodal support. Model + mmproj as separate GGUFs.
- **Quantization:** Hybrid per-layer Q5_K/Q6_K mixing ("Q6_K_H") —
  different quant levels per layer optimised for accuracy-size tradeoff.
- **Known issues:**
  - `[BEGIN_AUDIO]` token missing in llama.cpp's mtmd for Voxtral
    ([#17868](https://github.com/ggml-org/llama.cpp/issues/17868))
  - Audio truncation at 30s boundaries since b7410
    ([#18419](https://github.com/ggml-org/llama.cpp/issues/18419))
  - Community member reports "error rate is higher using llama.cpp at
    any quant level (even bf16) compared to running the full model
    using transformers or vllm"
  - Ollama dropped llama.cpp specifically for multimodal due to
    instability, considering a "complete ripup" needed for proper
    multimodal support

## Feature comparison

| Feature | **CrispASR** | **max-lt** | **llama.cpp mtmd** |
| --- | --- | --- | --- |
| Model files | **1 GGUF** | 2 (model + mmproj) | 2 (model + mmproj) |
| Mel compute | Baked in GGUF | llama.cpp handles | llama.cpp handles |
| Tokenizer | Embedded Tekken blob | llama.cpp native | llama.cpp native |
| Encoder graph | Hand-written ggml | llama.cpp mtmd | llama.cpp mtmd |
| LLM forward | Hand-written ggml | llama.cpp core | llama.cpp core |
| KV cache | F16, hand-managed | llama.cpp native | llama.cpp native |
| Flash attention | ✅ prefill + decode | ✅ (via llama.cpp) | ✅ (via llama.cpp) |
| GPU (Metal/CUDA) | ❌ (CPU-only now) | ✅ via llama.cpp | ✅ via llama.cpp |
| [BEGIN_AUDIO] bug | ✅ not affected | needs patch | needs manual fix |
| 30s truncation bug | ✅ not affected | affected (llama.cpp) | affected |
| llama.cpp server compat | ❌ | ❌ (standalone CLI) | ✅ |
| Ollama compat | ❌ | ❌ | partial (multimodal broken) |
| OpenAI API compat | ❌ | ❌ | ✅ via llama.cpp server |
| Microphone recording | ❌ | ✅ (miniaudio) | ❌ |
| Audio enhancement | ❌ | ✅ (HP filter, noise gate) | ❌ |
| Per-layer quant | ❌ (uniform Q4_K) | ❌ | ✅ (Q6_K_H hybrid) |
| Streaming decode | ❌ | ✅ (token-by-token) | ✅ |
| Diff-tested vs PyTorch | ✅ every stage | ❌ | ❌ |
| Lines of model code | ~1300 | ~100 (wrapper) | 0 (all in llama.cpp) |

## Correctness comparison

| | CrispASR | llama.cpp mtmd |
| --- | --- | --- |
| JFK transcript | ✓ correct | ✓ (with patch + padding) |
| WER vs transformers | not measured | "higher at any quant, even bf16" (user report) |
| [BEGIN_AUDIO] | handled natively | missing, needs patch |
| >30s audio | ✓ (encoder pads to 30s) | truncates at 30s boundaries |

The higher WER in llama.cpp is notable. A community member tested the
same bf16 model and found worse accuracy in llama.cpp than in
transformers/vLLM. This could be due to the mel computation, the
im2col/matmul paths, or subtle prompt construction differences. Our
CrispASR implementation was diff-tested against PyTorch at every
architectural boundary (LLM cosine sim 0.999973, top-5 5/5 match),
which gives confidence that our forward pass matches the reference.

## Performance comparison

Not directly benchmarked head-to-head (llama.cpp would need GPU to be
fair). On CPU at Q4_K:

| | CrispASR Q4_K (CPU 4-thread) | Notes |
| --- | --- | --- |
| jfk 11s | 70 s total, 242 ms/token | 3B model is heavy on CPU |
| GGUF size | 2.5 GB | |

For llama.cpp with GPU (Metal/CUDA), expect ~5-10× speedup on the LLM
forward, making it real-time or better. CrispASR's CPU-only path is
inherently slower for a 3B model.

## Strategic assessment: should we move towards mtmd compatibility?

### Current state of llama.cpp multimodal (mtmd)

The steampunque discussion reveals a concerning picture:
1. **Two open bugs** affecting Voxtral specifically ([#17868], [#18419])
   that have been ignored by maintainers
2. **Ollama dropped llama.cpp** for multimodal, citing instability and
   considering a "complete ripup" for proper multimodal support
3. A community member reports **worse accuracy** in llama.cpp than
   transformers/vLLM at the same precision
4. The maintainer of the existing Voxtral GGUF says "I will not debug
   it further since its a complete waste of time to post issues and just
   have them ignored"

This is a red flag for depending on llama.cpp's mtmd path for
production ASR workloads.

### Recommendation: hybrid approach

Keep both paths:

1. **CrispASR standalone (primary, for ASR):** Self-contained, no
   upstream bugs, diff-tested correctness, works today. Ship as the
   `voxtral-main` CLI + `cstr/voxtral-mini-3b-2507-GGUF` on HF.
   This is what users who want reliable transcription should use.

2. **Also produce llama.cpp-compatible GGUFs (secondary, for ecosystem):**
   Use llama.cpp's `convert_hf_to_gguf.py` to produce standard model +
   mmproj GGUFs. Upload alongside our custom GGUFs so that users who
   want llama.cpp server / Ollama / Open WebUI compatibility can use
   those. Document the [BEGIN_AUDIO] patch requirement and the 30s
   truncation workaround.

3. **Do NOT rewrite CrispASR to depend on llama.cpp's mtmd:** The mtmd
   API is unstable (per Ollama's assessment), the Voxtral support has
   two unfixed bugs, and the accuracy is worse. Our hand-written ggml
   graphs are the more reliable path for CPU ASR inference.

4. **Consider GPU support via ggml backends directly:** When we want
   GPU acceleration, use ggml's Metal/CUDA backends on our existing
   graph builders, not llama.cpp's abstraction layer. Our graphs
   already use `ggml_flash_attn_ext` which has GPU backend support.
   The main work would be wiring up `ggml_backend_metal_init()` or
   `ggml_backend_cuda_init()` as alternatives to `ggml_backend_cpu_init()`.

### What we'd gain from each approach

| Path | Gain | Cost |
| --- | --- | --- |
| Keep CrispASR standalone | Correctness, simplicity, no upstream deps | CPU-only for now |
| Add llama.cpp-compat GGUFs | Ecosystem access for power users | Convert + document, 2-file deploy |
| Rewrite to use mtmd API | GPU accel via llama.cpp, server compat | Inherit all mtmd bugs + instability |
| Add GPU via ggml backends | GPU accel WITHOUT mtmd dependency | ~50 LOC of backend init changes |

The 4th option (native ggml GPU) is the best long-term path. It gives
us Metal/CUDA without inheriting llama.cpp's multimodal instability.
The infrastructure is already there — our graphs use ggml ops that
have Metal/CUDA kernels.

## Also noted: max-lt's nice UX features

max-lt/voxtral-cpp has some UX features worth adopting:
- **Microphone recording** via miniaudio — enables live transcription
- **Audio enhancement** (DC removal, 80Hz highpass, noise gate,
  normalization) — improves quality on real-world recordings
- **Interactive model selector** — nice for first-time users

These are orthogonal to the inference backend and could be added to
CrispASR's CLI independently.
