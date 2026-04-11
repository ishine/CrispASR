"""Voxtral-Mini-4B-Realtime-2602 reference dump backend (stub).

Placeholder — the current dump path is `models/voxtral4b-dump-ref.py`,
which hooks the HF `VoxtralRealtimeForConditionalGeneration` forward to
capture mel, t_cond, ada_norm scales, and the generated transcript.

Fill this in by porting that script into the modular `dump()` interface.
The template is `tools/reference_backends/voxtral.py`; the 4B-specific
differences (audio padding, time_embedding delay_tokens=6, adaptive
RMSNorm, tied embeddings) all live in the PyTorch reference — the
backend adapter here just exposes them as named stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "t_cond",          # time embedding after delay_tokens=6
    "encoder_output",
    "projector_output",
    "llm_logits",
    "llm_argmax",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    raise NotImplementedError(
        "voxtral4b dump backend is a stub. Port models/voxtral4b-dump-ref.py "
        "into this module's dump() function — see tools/reference_backends/"
        "voxtral.py for a worked example of the interface.")
