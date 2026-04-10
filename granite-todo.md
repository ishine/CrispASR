> **STATUS (2026-04-10): NOT STARTED.** Granite Speech 4.0-1B evaluation pending.
> See TODO.md for priorities.

# ibm-granite/granite-4.0-1b-speech — port plan

## What it actually is

Not a dedicated ASR like Cohere/Parakeet/Canary. It's a **speech-LLM**: a
Conformer-CTC audio encoder feeds a BLIP-2 Q-former projector, which feeds
prompt tokens into a 1 B Granite causal LLM that produces the transcript
(or any other instruction-following text) via normal text generation.

## Architecture (from `config.json` + `model.safetensors.index.json`)

Total: **4.63 GB**, 954 tensors, three modules.

### 1. Audio encoder — `granite_speech_encoder` (Conformer-CTC-ish)
- 16 layers, hidden=1024, 8 heads, dim_head=128, FFN via `intermediate_size`
- conv kernel=15, input_dim=160 (80 logmels × 2 stacked frames),
  output_dim=348 (character CTC head — used at training, ignored at inference)
- context_size=200, max_pos_emb=512
- Tensor count: `input_linear` (2) + `layers` (528 = 16×33) + `out` (2) + `out_mid` (2)
- Architecturally close to our existing Parakeet/Canary FastConformer, but
  with a character-level CTC aux head instead of subword CTC. Should be a
  ~1 week port reusing the parakeet encoder graph.

### 2. Projector — BLIP-2 Q-former
- 2 transformer layers, hidden=1024, 16 heads, FFN=4096
- Learned `projector.query` tokens (1 tensor) → `qformer` (54 tensors)
  → `linear` (2 tensors, projects 1024 → 2048 to match LLM hidden)
- Outputs a fixed-length sequence of "soft audio tokens" that get
  injected into the Granite prompt at a placeholder position.
- Q-former is small but unusual for this repo — needs cross-attention from
  query tokens to encoder output. Maybe 3-5 days.

### 3. Language model — `ibm-granite/granite-4.0-1b-base`
- 40 layers, hidden=2048, 16 heads, **4 KV heads (GQA)**, FFN=4096
- vocab=100353, max_pos=4096, **RoPE**, RMSNorm, SiLU
- Granite "soft" multipliers: embedding_multiplier=12.0, logits_scaling=8.0,
  attention_multiplier=0.0078125, residual_multiplier=0.22
- 362 tensors. **This is the bulk of the parameters and the bulk of the work.**

## Strategic options

### Path A — bind to llama.cpp's existing Granite loader  ✅ recommended
llama.cpp already has full Granite support (loader + forward + RoPE + GQA +
the soft multipliers). We only need to port:
1. The audio encoder (reuse parakeet's FastConformer graph)
2. The Q-former projector
3. A glue layer that injects projector outputs as input embeddings into a
   llama.cpp Granite context at a placeholder token offset

Pros: avoids re-implementing 40 transformer layers + RoPE + GQA + the
Granite multipliers. Picks up llama.cpp's quantisation, sampling, KV
cache, batching, and SIMD kernels for free. Granite-1B already has
solid llama.cpp coverage.

Cons: introduces a llama.cpp dependency (or vendored subset). The
"speech tokens injection" hook requires either patching llama.cpp's
`llama_decode` to accept pre-computed embeddings, or using the existing
`llama_get_embeddings` / `llama_set_embeddings` API if available.

Estimated effort: **2-3 weeks**.

### Path B — fully standalone Granite forward in this repo
Write a slim Granite-1B forward in `src/granite_llm.{h,cpp}` matching
what llama.cpp does, with the soft multipliers baked in.

Pros: no external runtime dependency, single binary, full control over
the speech-token injection path.

Cons: re-implements ~3000 lines of well-tested llama.cpp code. Easy to
get GQA / RoPE / multiplier order wrong. Quantisation work doubles
(need to verify Q4_K / Q8_0 paths against llama.cpp output). Slower
inference than llama.cpp's tuned kernels.

Estimated effort: **5-6 weeks**.

## Recommendation

**Path A.** The audio encoder + projector are the interesting / novel
work and play to this repo's strengths. The LLM forward is a solved
problem and llama.cpp already does it better than we would. Vendor
llama.cpp as a subtree (same way we vendor ggml), expose its Granite
loader, and write a thin `granite-speech-main` CLI that:

1. Loads the audio encoder + Q-former from `granite-speech-encoder.gguf`
   (our conversion)
2. Loads the LLM from `granite-1b.gguf` (standard llama.cpp Granite GGUF —
   may already exist on HF, otherwise convert via llama.cpp's
   `convert_hf_to_gguf.py`)
3. Computes mel → encoder → Q-former → soft audio tokens
4. Builds a prompt: `<|system|>...<|user|><audio_placeholder>Transcribe<|assistant|>`
5. Calls llama.cpp's decode loop, splicing soft tokens in at the
   placeholder offset

## Open questions before starting

- **Does HF have a llama.cpp-compatible Granite-1B GGUF already?** If
  yes, we only need to ship the encoder+projector GGUF (~500 MB) instead
  of bundling the whole 4.6 GB.
- **What does the chat template / audio placeholder token look like?**
  Need to read `processor_config.json` and `chat_template.json` (not yet
  downloaded) to find the exact prompt format.
- **Is the BLIP-2 Q-former cross-attention standard?** Need to inspect
  the actual tensor names under `projector.qformer.*` to confirm it's
  the BERT-style Q-former and not something custom.
- **Licensing.** Granite is Apache-2.0 — fine to redistribute the
  encoder GGUF on HF.

## Decision needed from user

Path A or Path B? And: do we want to gate this on first finishing the
upstream ffmpeg-transcode fix (UPSTREAM.md), or start in parallel?
