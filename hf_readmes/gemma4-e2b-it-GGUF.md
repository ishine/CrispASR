---
license: apache-2.0
language:
- en
- multilingual
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- gguf
- gemma
- conformer
library_name: ggml
base_model: google/gemma-4-E2B-it
---

# Gemma-4-E2B-it -- GGUF

GGUF conversion of [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `gemma4-e2b-it.gguf` | F16 | 9.5 GB | Full precision (872 tensors) |
| `gemma4-e2b-it-q4_k.gguf` | Q4_K | ~2.5 GB | Quantized (pending) |

## Model details

- **Architecture:** USM Conformer audio encoder (12L, 1024d, chunked attention, LightConv1d) + Gemma4 LLM decoder (35L, 1536d, GQA 8Q/1KV, per-layer embeddings, hybrid sliding/full attention, SwiGLU)
- **Parameters:** 2.3B effective (5.1B with embeddings)
- **Audio:** 128-bin log-mel, 16kHz, 40ms frames, max 30 seconds
- **Languages:** 140+ (ASR + speech translation)
- **License:** Apache 2.0
- **Source:** [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)

## Notes

- This GGUF includes the **audio conformer encoder** (872 tensors) — standard Gemma-4 GGUFs (unsloth, ggml-org) only have text+vision (601 tensors) and cannot do ASR
- CrispASR runtime implementation in progress
- Vision tower tensors and clipped-linear min/max scalars are excluded

## Usage with CrispASR

```bash
./build/bin/crispasr --backend gemma4-e2b -m gemma4-e2b-it-q4_k.gguf -f audio.wav
```
