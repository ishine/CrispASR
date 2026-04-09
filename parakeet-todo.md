# Parakeet TDT 0.6B v3 — ggml port plan

> **STATUS (2026-04-09): ✅ MOSTLY COMPLETE.** Encoder (ggml graph), TDT
> decoder (CPU), word timestamps, --flash, GPU auto-detect, HF release
> shipped. Remaining: decoder LSTM+Joint still runs raw CPU float* loops
> (see TODO.md P1). See `TODO.md` for current task list.

Goal: a `parakeet-main` CLI in `whisper.cpp` that runs `nvidia/parakeet-tdt-0.6b-v3`
on CPU via ggml, with built-in word-level timestamps.

## Why

- **Word timestamps come for free** from the TDT decoder's duration head — no separate CTC alignment model needed (unlike `cohere-align`).
- **600 M params (vs 2 B for Cohere)** → ~400 MB at Q4_K, ~3× faster.
- **25 EU languages** with automatic language detection (no prompt prefix).
- **CC-BY-4.0** licence.
- **No existing ggml port.** This is novel work.
- The encoder is the same Conformer family that `cohere.cpp` already implements,
  so ~80 % of the encoder code can be reused.

## Architecture summary

```
audio → mel(80, 16 kHz) → FastConformer encoder → encoder_out[T, H]
                                                       │
                                                       ▼
                              ┌─── joiner[T, U, V+1] ──── argmax → (token, duration)
                              │           ▲
                              │           │
                              │   predictor (LSTM)
                              │           ▲
                              │           │
                              └────  emitted tokens
```

- **Encoder:** FastConformer, 24 layers, hidden 1024, subsampling 8× (Conv2d
  striding to 8× downsample over time). Output frame rate: 12.5 Hz (~80 ms / frame).
- **Predictor:** stateful 1- or 2-layer LSTM over the SentencePiece token stream
  (8 192 vocab). Hidden ~640.
- **Joiner:** small MLP `(enc[t] + pred[u]) → joiner_hidden → V+1` where V = 8 192
  and the +1 is the blank token. **TDT variant** also outputs a duration over
  `{0, 1, 2, 3, 4}` (skip-frames-after-emit).
- **Greedy TDT decode loop:** at each `(t, u)` step, predict (token, duration);
  if token == blank, advance t; otherwise emit token, advance u and t by `duration`.
- **Timestamps:** the `duration` already encodes how many encoder frames each
  token spans. `t * 0.08 s` is the start time, `(t + duration) * 0.08 s` is the end.

## Tasks

### 1. Download + inspect .nemo (1 day)

- [ ] Download `nvidia/parakeet-tdt-0.6b-v3` from HF
- [ ] Unpack the `.nemo` tarball, dump `model_config.yaml` and the keys of
      `model_weights.ckpt`
- [ ] Verify the encoder really is FastConformer and confirm hyperparams
- [ ] Document tensor naming conventions (NeMo uses `encoder.encoder.0.norm_self_att.weight` etc.)

### 2. .nemo → GGUF conversion script (2-3 days)

- [ ] `models/convert-parakeet-to-gguf.py` mirroring `convert-wav2vec2-to-gguf.py`
- [ ] Map NeMo tensor names to a flat ggml-friendly schema
- [ ] Convert encoder, predictor LSTM, and joiner
- [ ] Bake mel filterbank into GGUF metadata (or compute at runtime as in `cohere.cpp`)
- [ ] Embed SentencePiece vocab in GGUF metadata under `tokenizer.ggml.tokens`
- [ ] Write hyperparams under the `parakeet.*` namespace
- [ ] Test loading from C++ side (just open the file, list tensors)

### 3. Refactor Conformer encoder out of cohere.cpp (2-3 days)

- [ ] Extract `conformer_block_forward()` and supporting helpers from
      `cohere.cpp` into `src/conformer.{h,cpp}` (a header-only template would
      be cleaner if the layer counts and dims differ)
- [ ] Make the existing `cohere.cpp` use the extracted version (no behaviour
      change, regression tests on cohere-main pass)
- [ ] Reuse mel filterbank code unchanged

### 4. FastConformer-specific tweaks (1-2 days)

- [ ] Subsampling: Cohere uses Conv1d, FastConformer uses Conv2d → re-implement
      the subsampling stack (8× over time, 80→hidden over freq)
- [ ] Limited-context attention (`rel_pos_local_attn` window mask) for the
      long-form mode — optional for v1, can default to full attention
- [ ] Rotary or relative positional encoding — match NeMo's choice (FastConformer
      uses rel-pos by default)

### 5. Predictor LSTM (3-4 days)

- [ ] New file `src/parakeet.cpp` with TDT-specific code
- [ ] LSTM forward pass via ggml (`ggml_lstm` exists in some forks; otherwise
      manual gate computation — only ~60 LOC)
- [ ] State management struct: `predictor_state { h, c }` per utterance
- [ ] Token embedding lookup → LSTM step → predictor_out

### 6. Joiner + greedy TDT decode loop (3-4 days)

- [ ] Joiner forward: `pred + enc → tanh → linear → V+1+5` (logits over
      vocab+blank, plus 5-class duration head)
- [ ] Greedy loop: walk t from 0 to T_enc, at each t emit tokens until blank,
      advancing u and t per the duration prediction
- [ ] Detokenisation via SentencePiece (`▁` → space)

### 7. Word and segment timestamps (1-2 days)

- [ ] During the decode loop, record `(token_id, t_start, t_end)` for every
      emitted token where `t_start = t * frame_dur`, `t_end = (t + duration) * frame_dur`
- [ ] Group sub-word tokens into words (at `▁` boundaries)
- [ ] Group words into segments (at punctuation or VAD boundaries)
- [ ] Mirror `cohere_token_data` and `cohere_result` structs as
      `parakeet_token_data` / `parakeet_result`

### 8. CLI tool `parakeet-main` (2 days)

- [ ] `examples/parakeet-main/main.cpp` mirroring `cohere-main.cpp` API
      (`-m`, `-f`, `-l`, `-ts`, `-osrt`, `-ovtt`, `-vad-model`, …)
- [ ] No `-l` required (auto language detection from the model itself)
- [ ] CMakeLists.txt in `examples/parakeet-main/`
- [ ] Register in `examples/CMakeLists.txt`

### 9. CMake library wiring (1 day)

- [ ] `add_library(parakeet ...)` in `src/CMakeLists.txt`
- [ ] Same BLAS / Metal / CUDA conditional setup as the cohere library
- [ ] Reuse the wav2vec2-ggml + ctc-align libraries unchanged
- [ ] Link `parakeet-main` against `parakeet`, `whisper`, `common`

### 10. Testing (2-3 days)

- [ ] Sanity check on `samples/jfk.wav` — should produce "And so my fellow
      Americans..." with reasonable word timestamps
- [ ] Compare against the NeMo Python reference on the same audio
- [ ] WER spot check on a few clips from each of the 25 supported languages
- [ ] Quantise to Q4_K, Q5_0, Q8_0 — confirm no quality regression
- [ ] Benchmark vs Cohere on the voxpopuli demo clip

### 11. Documentation (1 day)

- [ ] Add a `parakeet-main` section to `README.md`
- [ ] Document the model conversion: `python models/convert-parakeet-to-gguf.py`
- [ ] Document timestamp accuracy: should be ~80 ms (one encoder frame) which
      is comparable to `cohere-align` and *much* better than `cohere-main`'s
      cross-attention DTW
- [ ] Update `benchmark_cohere.md` with parakeet rows

## Total estimate: ~3 weeks

For a working CPU runtime with word timestamps. Q4_K quantisation will land
the model around 400 MB, so a ~5 s clip should run in 3-5 s on the same box
where Cohere Q4_K runs in 15 s.

## Out of scope for v1

- Streaming / chunked inference (NeMo has it; we'd inherit it later)
- Beam search (greedy is fine for TDT)
- Speaker diarisation (no model support)
- The 24-minute long-audio mode (`rel_pos_local_attn`) — can be added in v2

## Reference implementations to crib from

- **`istupakov/onnx-asr`** — Python ONNX inference loop for Parakeet TDT.
  Best reference for the joiner + greedy decode logic.
- **NeMo** `nemo/collections/asr/modules/rnnt.py` — the ground-truth implementation.
- **`cohere.cpp`** — for the Conformer encoder block, mel filterbank, and
  CMake/library wiring patterns.
- **`wav2vec2-ggml.cpp`** — for the manual ggml inference loop pattern.
