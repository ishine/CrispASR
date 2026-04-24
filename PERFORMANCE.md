# CrispASR — Performance Benchmarks

All benchmarks on jfk.wav, 4 threads, CPU-only (no GPU), Q4_K quant.
Machine: x86_64, 7.6 GB RAM, clang-format 18, AVX2.

Date: 2026-04-24

## Summary (11s audio)

| # | Backend | Model | RT factor | Time | Notes |
|---|---|---|---|---|---|
| 1 | **moonshine** | moonshine-tiny | **16.8x** | 0.7s | Tiny (27M), English only |
| 2 | **fc-ctc** | stt-en-fc-ctc-large | **9.4x** | 1.2s | NeMo CTC, English only |
| 3 | **parakeet** | parakeet-tdt-0.6b-v3 | **2.9x** | 3.8s | TDT, 25 EU languages |
| 4 | **canary** | canary-1b-v2 | **2.7x** | 4.0s | Seq2seq, 25 EU languages |
| 5 | **data2vec** | data2vec-audio-base | **2.1x** | 5.2s | CTC, English only |
| 6 | **qwen3** | Qwen3-ASR-0.6B | **1.7x** | 6.5s | LLM, 30+ languages |
| 7 | **omniasr-300m** | omniASR-CTC-300M | **1.4x** | 7.7s | CTC, 1600+ languages |
| 8 | **cohere** | cohere-transcribe | **1.4x** | 7.7s | Seq2seq, 13 languages |
| 9 | **wav2vec2** | wav2vec2-large-xlsr | **1.1x** | 9.9s | CTC, per-model language |
| 10 | **omniasr-1b** | omniASR-CTC-1B | **0.7x** | 15.7s | CTC, 1600+ languages |
| 11 | **omniasr-llm** | omniASR-LLM-300M-v2 | **0.4x** | 29.2s | LLM, 1600+ languages |
| 12 | **firered** | FireRedASR2-AED | **0.1x** | 123.0s | CPU decoder bottleneck |

## Long audio scaling (55s = 5× the 11s clip)

| Backend | 11s | 55s | Scale | Notes |
|---|---|---|---|---|
| moonshine | 0.7s | 5.8s | 8.3x | Superlinear (decoder grows with output) |
| fc-ctc | 1.2s | 5.2s | 4.3x | Sublinear (efficient encoder) |
| parakeet | 3.8s | 18.2s | 4.8x | Near-linear |
| canary | 4.0s | 20.7s | 5.2x | Near-linear |
| data2vec | 5.2s | 25.9s | 5.0x | Linear |
| qwen3 | 6.5s | 29.8s | 4.6x | Good |
| wav2vec2 | 9.9s | 57.4s | 5.8x | Slight superlinear (pos_conv O(T)) |
| omniasr-300m | 7.7s | 37.5s | 4.9x | Linear |
| omniasr-1b | 15.7s | 80.8s | 5.1x | Linear |
| cohere | 7.7s | 41.5s | 5.4x | Near-linear |
| firered | 123s | **TIMEOUT** | — | CPU decoder, needs ggml graph |

Auto-chunking at 30s keeps all backends memory-bounded. No OOM on 55s.

## Per-phase breakdown (wav2vec2 family)

| Model | CNN | Pos conv | Encoder | Total |
|---|---|---|---|---|
| wav2vec2-large (24L, d=1024) | 2.3s | 2.1s | 6.1s | 9.9s |
| hubert-large (24L, d=1024) | 2.3s | 1.9s | 6.3s | 10.6s |
| data2vec-base (12L, d=768) | 2.1s | 0.9s | 2.5s | 5.2s |

## Optimization history

### wav2vec2 CNN → ggml (10.8x speedup)

| Change | CNN | Pos conv | Total | Speedup |
|---|---|---|---|---|
| Baseline (manual C++) | 95.2s | 6.8s | 108.4s | 1.0x |
| ggml F32 im2col | 2.4s | 6.8s | 15.5s | 7.0x |
| + OpenMP pos_conv | 2.3s | 2.1s | 9.9s | **10.9x** |

## Optimization roadmap

### From koboldcpp / llama.cpp patterns

| Optimization | Applicable to | Expected gain | Effort | Status |
|---|---|---|---|---|
| **KV cache Q8_0** | All LLM backends (7) | 2× less KV memory | Low | TODO |
| **Encoder output caching** | Server mode | Skip re-encode for same audio | Low | TODO |
| **Multi-threaded mel** | All mel-based backends (8) | <5% (mel is <100ms) | Low | Low priority |
| **Speculative decoding** | LLM backends | 2-4× decode speed | High | TODO |
| **GPU layer autofit** | All backends | Auto GPU/CPU split | Medium | TODO |
| **Batched encoder** | All (GPU only) | 3-5× on GPU | High | TODO |

### CrispASR-specific

| Optimization | Applicable to | Expected gain | Effort | Status |
|---|---|---|---|---|
| **FireRed decoder → ggml graph** | firered-asr | ~10× (123s→~12s) | High | Critical |
| **Fused QKV pre-merge** | LLM decoders | ~10-15% attention | Medium | TODO |
| **wav2vec2 pos_conv → ggml** | wav2vec2 family | ~1.5× (2.1s→~0.7s) | Medium | TODO |
| **Temperature sampling** | 5 backends | Feature parity | Low | TODO |

### Priority: FireRed decoder

FireRed-ASR at 0.1x RT is the worst performer. The encoder is fast (ggml graph)
but the decoder runs manual C++ loops for self-attention, cross-attention, and
MLP at each decode step. Moving the heavy decoder matmuls to persistent ggml
subgraphs (as documented in LEARNINGS.md) is the highest-impact remaining
optimization.

## Reproduce

```bash
# Per-phase timing
WAV2VEC2_BENCH=1 crispasr --backend wav2vec2 -m model.gguf -f audio.wav -l en
OMNIASR_BENCH=1 crispasr --backend omniasr -m model.gguf -f audio.wav -l en

# Full benchmark (all backends, 11s + 55s)
bash tools/benchmark_phases.sh 4

# Memory + verbose
crispasr --backend wav2vec2 -m model.gguf -f audio.wav -v
```
