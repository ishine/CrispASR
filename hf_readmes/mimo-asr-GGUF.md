---
license: mit
language:
- zh
- en
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- gguf
- mimo
- qwen2
library_name: ggml
base_model: XiaomiMiMo/MiMo-V2.5-ASR
---

# MiMo-V2.5-ASR -- GGUF

GGUF conversion of [`XiaomiMiMo/MiMo-V2.5-ASR`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `mimo-asr.gguf` | F16 | 15.3 GB | Full precision (719 tensors) |
| `mimo-asr-q4_k.gguf` | Q4_K | ~4.5 GB | Quantized (pending) |

## Model details

- **Architecture:** 8-channel RVQ speech embeddings + 6-layer input transformer (1024d, 64 heads) + 36-layer Qwen2 LLM (4096d, 32 heads, 8 KV heads, SiLU, RoPE θ=640K)
- **Parameters:** ~8B
- **Languages:** Mandarin Chinese, English, Chinese dialects (Wu, Cantonese, Hokkien, Sichuanese), code-switched speech
- **License:** MIT
- **Source:** [`XiaomiMiMo/MiMo-V2.5-ASR`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR)

## Notes

- Requires separate `MiMo-Audio-Tokenizer` to convert waveform → RVQ tokens first
- CrispASR runtime implementation in progress

## Usage with CrispASR

```bash
./build/bin/crispasr --backend mimo-asr -m mimo-asr-q4_k.gguf -f audio.wav
```
