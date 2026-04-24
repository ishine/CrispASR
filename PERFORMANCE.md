# CrispASR — Performance Benchmarks

All benchmarks on jfk.wav (11.0s, 16kHz mono), 4 threads, CPU-only
(no GPU), Q4_K quantization where available. Machine: x86_64, 7.6 GB RAM.

Date: 2026-04-24

## Summary table

| Backend | Model | Params | Size (Q4_K) | Time (s) | Realtime | Notes |
|---|---|---|---|---|---|---|
| **parakeet** | parakeet-tdt-0.6b-v3 | 600M | 466 MB | **4.2** | **2.6x** | FastConformer + TDT |
| **canary** | canary-1b-v2 | 1B | 673 MB | **4.4** | **2.5x** | FastConformer + Transformer dec |
| **data2vec** | data2vec-audio-base-960h | 95M | 79 MB | **5.6** | **2.0x** | 12L, d=768 |
| **qwen3** | Qwen3-ASR-0.6B | 600M | 513 MB | **5.9** | **1.9x** | Whisper enc + Qwen3 LLM |
| **wav2vec2-base** | wav2vec2-base-voxpopuli-it | 95M | 81 MB | **6.0** | **1.8x** | 12L, d=768 |
| **omniasr-300m** | omniASR-CTC-300M | 300M | 194 MB | **7.3** | **1.5x** | 24L, d=1024, ggml graph |
| **cohere** | cohere-transcribe | 2B | 1.5 GB | **9.4** | **1.2x** | Conformer + Transformer dec |
| **hubert** | hubert-large-ls960-ft | 316M | 212 MB | **10.6** | **1.0x** | 24L, d=1024 |
| **wav2vec2** | wav2vec2-large-xlsr-53-en | 315M | 212 MB | **11.1** | **1.0x** | 24L, d=1024 |
| **omniasr-1b** | omniASR-CTC-1B | 1B | 551 MB | **17.6** | **0.6x** | 48L, d=1280 |
| **omniasr-llm** | omniASR-LLM-300M-v2 | 1.5B | 1.1 GB | **29.2** | **0.4x** | Enc + LLaMA decoder |

## Per-phase analysis: wav2vec2 family

All wav2vec2-family models (wav2vec2, data2vec, HuBERT) share the same pipeline:

```
raw PCM → CNN (7 layers) → pos_conv → transformer encoder → LM head → CTC decode
```

| Model | CNN | Pos conv | Encoder | Total |
|---|---|---|---|---|
| wav2vec2-large (24L, d=1024) | 2.3s | 2.1s | 6.7s | 11.1s |
| hubert-large (24L, d=1024) | 2.3s | 1.9s | 6.3s | 10.6s |
| wav2vec2-base (12L, d=768) | 2.6s | 1.3s | 2.1s | 6.0s |
| data2vec-base (12L, d=768) | 2.1s | 0.9s | 2.5s | 5.6s |

### Optimization history (wav2vec2-large)

| Change | CNN | Pos conv | Encoder | Total | Cumul speedup |
|---|---|---|---|---|---|
| Baseline (manual C++) | 95.2s | 6.8s | 6.3s | 108.4s | 1.0x |
| ggml F32 im2col CNN | 2.4s | 6.8s | 6.3s | 15.5s | 7.0x |
| + OpenMP pos_conv | 2.3s | 2.1s | 6.7s | 11.1s | **9.8x** |

## Per-phase analysis: OmniASR family

Fully on ggml graphs. Scales linearly with layer count:

| Model | Layers | d_model | Enc compute | Total |
|---|---|---|---|---|
| CTC-300M | 24 | 1024 | 7.2s | 7.3s |
| CTC-1B | 48 | 1280 | 17.0s | 17.6s |
| LLM-300M | 24+12 | 1024+4096 | 7.2s+12s | 29.2s |

## Reproduce

```bash
# Per-backend with timing
WAV2VEC2_BENCH=1 crispasr --backend wav2vec2 -m model.gguf -f samples/jfk.wav -l en
OMNIASR_BENCH=1 crispasr --backend omniasr -m model.gguf -f samples/jfk.wav -l en

# All backends
bash tools/benchmark_all.sh 4
```
