"""NeMo Parakeet-TDT reference dump backend.

Loads `nvidia/parakeet-tdt_ctc-0.6b-ja` (or v3) via NeMo and captures the
mel features and final encoder output for crispasr-diff comparison
against the C++ runtime. Intended for diagnosing the JA TDT decoder bug
(emits 1 token then collapses to blanks) where the C++ encoder
diverged from the NeMo reference at the FIRST output frame.

Stages:

  raw_audio        (N,)            input PCM
  mel_spectrogram  (n_mels, T_mel) NeMo preprocessor output, batch-stripped
  encoder_output   (T_enc, d_model) FastConformer encoder output

Captures match the names that `examples/cli/crispasr_diff_main.cpp`
already looks up for the "parakeet" backend.

Usage:

  python tools/dump_reference.py --backend parakeet \\
      --model-dir nvidia/parakeet-tdt_ctc-0.6b-ja \\
      --audio /tmp/parakeet-ja/ja16k.wav \\
      --output /tmp/parakeet-ja/ref.gguf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_output",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run NeMo Parakeet-TDT reference forward and return stage captures.

    `model_dir` may be either a local path to an extracted .nemo or a
    HuggingFace pretrained name (e.g. "nvidia/parakeet-tdt_ctc-0.6b-ja").
    """
    import torch
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise SystemExit(
            "NeMo toolkit required.\n"
            "Install: pip install 'nemo_toolkit[asr]'\n"
            f"(import error: {e})")

    pretrained = str(model_dir)
    print(f"  loading NeMo Parakeet-TDT model from {pretrained}")
    if pretrained.startswith("nvidia/") or "/" not in pretrained:
        model = nemo_asr.models.ASRModel.from_pretrained(pretrained)
    else:
        model = nemo_asr.models.ASRModel.restore_from(pretrained)
    model.eval()
    dev = next(model.parameters()).device

    sig = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
    sig_len = torch.tensor([audio.shape[0]], device=dev)

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    with torch.no_grad():
        feats, feat_len = model.preprocessor(input_signal=sig, length=sig_len)
        # feats: (B=1, n_mels, T_mel). The C++ runtime stores mel in the
        # TimeMels layout (T_mel, n_mels) — ne[0]=n_mels is the fast axis.
        # Transpose so flat-element ordering matches what
        # parakeet_compute_mel returns.
        if "mel_spectrogram" in stages:
            m = feats[0].transpose(0, 1).contiguous()
            out["mel_spectrogram"] = m.detach().cpu().float().numpy()

        enc, enc_len = model.encoder(audio_signal=feats, length=feat_len)
        # enc: (B=1, d_model, T_enc) in NeMo's convention. crispasr-diff's
        # parakeet_encoder_r returns (T_enc, d_model), so transpose to match.
        if "encoder_output" in stages:
            T_enc = int(enc_len.item())
            e = enc[0, :, :T_enc].transpose(0, 1).contiguous()
            out["encoder_output"] = e.detach().cpu().float().numpy()

    return out
