---
license: apache-2.0
language:
- en
- fr
- de
- es
- pt
- ja
base_model:
- ibm-granite/granite-speech-4.1-2b-plus
tags:
- asr
- speech
- gguf
- crispasr
- granite-speech-plus
- speaker-attributed
- word-timestamps
---

# granite-speech-4.1-2b-plus — GGUF

GGUF conversion of [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)
for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

The PLUS variant adds two capabilities over the base 4.1-2b:

- **Punctuated and capitalised transcripts by default** — no special
  prompt required.
- **Speaker labels and word-level timestamps** in the model's structured
  output (full output parsing in CrispASR is the next step; raw text
  works today).

Architecturally PLUS is the base 4.1-2b plus a single change: the
encoder's layer-3 hidden state is concatenated with the final layer
output (config: `cat_hidden_layers: [3]`), producing a 2048-dim
projector input instead of 1024. The Q-Former cross-attention K/V
projection weights are correspondingly `(1024, 2048)`.

## Files

| File | Quantisation | Size | Notes |
|---|---|---|---|
| `granite-speech-4.1-2b-plus-f16.gguf` | F16 | ~5.6 GB | Encoder + projector in F32, LLM weights in F16 |

Q4_K and Q4_K-f16enc variants will land once parity is validated.

## Usage with CrispASR

```bash
# auto-download and transcribe
crispasr --backend granite-4.1-plus -m auto samples/audio.wav

# or with explicit path
crispasr --backend granite-4.1-plus \
  -m granite-speech-4.1-2b-plus-f16.gguf \
  samples/audio.wav
```

End-to-end example on the JFK 11s clip:

```
$ crispasr --backend granite-4.1-plus -m auto samples/jfk.wav
And so my fellow Americans, ask not what your country can do for
you, ask what you can do for your country.
```

(Note the punctuation + capitalisation that the base 4.1-2b only
produces with an explicit `--ask "transcribe with proper
punctuation..."` prompt.)

## Architecture

| Stage | Description |
|---|---|
| Encoder | 16-layer Macaron Conformer (1024 dim, 8 heads, 15-tap depthwise conv). Hidden state at layer 3 is captured and concatenated with the final layer output → 2048-dim projector input. |
| Projector | 2-layer BLIP-2 Q-Former. Cross-attention K/V weights are `(1024, 2048)` to consume the wider concatenated encoder feature. 3 learned query tokens per 15-frame window. |
| LLM | Granite 4.0-1B (40 layers, 2048 hidden, GQA 16/4, SwiGLU, RoPE θ=10000, μP multipliers). |

Total ~2.2 B parameters. The "+" capability is encoded entirely in
training — the architectural delta from base is just the layer
concatenation.

## Conversion

```bash
python models/convert-granite-speech-to-gguf.py \
  --input /path/to/granite-speech-4.1-2b-plus \
  --output granite-speech-4.1-2b-plus-f16.gguf
```

The same converter handles base / 4.1-2b / 4.1-2b-plus from a single
script — variant detection happens via `config.json` keys
(`cat_hidden_layers`, `encoder_hidden_size`).

## Licence

Apache 2.0 — same as the original
[ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus).
