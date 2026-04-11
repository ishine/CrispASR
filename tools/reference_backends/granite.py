"""Granite Speech 4.0-1B reference dump backend (stub).

Placeholder — the current dump path is
`models/granite-speech-kaggle-groundtruth.py`, a Kaggle-specific script
that requires HF_TOKEN and produces its own .npy dumps.

Port it into the modular interface by following the pattern in
`tools/reference_backends/qwen3.py`. Granite-specific stages of interest:

  mel_spectrogram   (1, 160, T/2) F32  — stacked 2-frame mel
  encoder_output    (1, T_enc, 1024) F32  — after 16-layer Conformer
  qformer_queries   (1, 3, 1024) F32    — BLIP-2 Q-Former learned queries
  projector_output  (1, 3, 2048) F32    — after linear 1024 -> 2048
  llm_logits        (1, T, 100353) F32  — Granite 1B LLM output
  llm_argmax        (1, T) int32        — greedy token IDs

The µP scale multipliers (embedding_multiplier=12.0,
attention_multiplier=1/128, residual_multiplier=0.22, logits_scaling=8.0)
are baked into the PyTorch reference; the dumper just captures the
post-scale activations via hooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_output",
    "qformer_queries",
    "projector_output",
    "llm_logits",
    "llm_argmax",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    raise NotImplementedError(
        "granite dump backend is a stub. Port "
        "models/granite-speech-kaggle-groundtruth.py into this module's "
        "dump() function — see tools/reference_backends/qwen3.py for a "
        "worked example of the interface.")
