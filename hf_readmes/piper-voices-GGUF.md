---
license: cc0-1.0
language:
- de
- en
base_model:
- rhasspy/piper-voices
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- piper
- vits
- gguf
- crispasr
library_name: ggml
---

# Piper voices — GGUF bundle

[rhasspy/piper](https://github.com/rhasspy/piper) VITS voices converted to
ggml's GGUF format (arch=`piper`) for the **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**
native runtime. Each voice is a single self-contained file — the
phoneme-id map and the espeak-ng voice are embedded in the GGUF, so no
companion is needed. Output is 22.05 kHz mono (CrispASR resamples to its
24 kHz playback convention).

Tiny and fast: ~15–60 MB per voice, single-digit-ms-per-sentence on CPU —
the best fit for mobile (CrisperWeaver) and quick previews. Converted with
[`models/convert-piper-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-piper-to-gguf.py)
(F16; reads the upstream `.onnx` + `.onnx.json`).

## Voices

Only **permissively-licensed** voices are hosted here — the underlying
training datasets allow redistribution (CC0 / public domain). Restrictive
voices from `rhasspy/piper-voices` (e.g. `en_US-lessac`, Blizzard 2013
research license; the CC BY-NC-SA voices) are **deliberately excluded**.

| File | Voice | Language | Quality | Dataset license |
|---|---|---|---|---|
| `piper-de_DE-thorsten-medium-f16.gguf` | thorsten | German (M) | medium | **CC0** |
| `piper-de_DE-thorsten-high-f16.gguf` | thorsten | German (M) | high | **CC0** |
| `piper-de_DE-thorsten_emotional-medium-f16.gguf` | thorsten (emotional) | German (M) | medium | **CC0** |
| `piper-en_GB-cori-medium-f16.gguf` | cori | English (GB, F) | medium | **public domain** |

The German voices are [Thorsten-Voice](https://www.thorsten-voice.de/)
(Thorsten Müller), released **CC0** (public domain dedication). `en_GB-cori`
is released into the **public domain** per its upstream MODEL_CARD.

## Licensing

Two layers, both permissive here:

- **Runtime + converter** — the Piper architecture, the espeak-ng
  phonemizer integration, and `convert-piper-to-gguf.py` are MIT
  (rhasspy/piper is MIT-licensed; CrispASR's runtime is its own).
- **Voice weights** — each GGUF is a derivative of an upstream Piper voice
  and carries **that voice's dataset license** (the table above). Every
  voice hosted here is CC0 or public domain, so the GGUFs are free to
  use, redistribute, and ship commercially. No attribution is legally
  required, but crediting [Thorsten-Voice](https://www.thorsten-voice.de/)
  and the [Piper](https://github.com/rhasspy/piper) project is appreciated.

To add a voice with a different license (e.g. the CC BY 4.0
`en_US-libritts_r`), convert it yourself and honour that voice's terms —
do not assume the repo-level tag applies.

## Usage

```bash
crispasr --backend piper -m piper-de_DE-thorsten-medium-f16.gguf \
  --tts "Guten Tag, dies ist ein Test." --tts-output out.wav
```

In CrisperWeaver these appear in the Synthesize screen's model picker once
downloaded; the backend resamples 22.05 kHz → 24 kHz transparently.
