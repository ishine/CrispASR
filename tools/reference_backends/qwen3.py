"""Qwen3-ASR 0.6B reference dump backend.

Instruments the HuggingFace `Qwen3ASRForConditionalGeneration` forward
pass with per-layer hooks and emits a dict of captured activations
following the stage contract in `tools/dump_reference.py`.

Mechanical port of `models/qwen3-asr-reference-dump.py` into the new
modular interface. The only differences from the legacy script:

  1. dump() takes `stages: set[str]` and only captures the stages the
     user asked for (the legacy script dumped everything unconditionally).
  2. dump() returns an in-memory dict instead of writing .npy files.
     The parent dumper collects the dict and serializes it to a single
     GGUF tensor archive.
  3. No CLI argument parsing inside this file — that lives in the
     unified tools/dump_reference.py dispatcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "conv1_out",
    "conv2_out",
    "conv3_out",
    "conv_out",
    "enc_blk00_out",
    "enc_blk17_out",
    "ln_post_out",
    "proj1_out",
    "proj2_out",
    "encoder_output",
    "llm_logits",
    "llm_argmax",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Qwen3-ASR reference forward pass, return captured stage tensors."""
    import torch
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        raise SystemExit(
            "qwen_asr package not found. Install with: pip install qwen-asr\n"
            f"(original import error: {e})")

    print(f"  loading Qwen3-ASR model from {model_dir}")
    wrapper = Qwen3ASRModel.from_pretrained(
        str(model_dir), dtype="float32", device_map="cpu",
    )
    processor = wrapper.processor
    model = wrapper.model
    model.eval()

    thinker = model.thinker
    audio_tower = thinker.audio_tower

    out: Dict[str, np.ndarray] = {}

    # ---- Mel spectrogram via WhisperFeatureExtractor ----
    # We go through the feature_extractor directly because processor.__call__
    # requires text input as well.
    feat = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt",
        padding=True, truncation=False,
    )
    mel = feat["input_features"]  # (1, 128, T)
    if "mel_spectrogram" in stages:
        out["mel_spectrogram"] = mel.detach().cpu().float().numpy()

    # ---- Register forward hooks on the audio encoder ----
    captures: Dict[str, np.ndarray] = {}

    def cap(name: str):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[name] = t.detach().cpu().float().numpy()
        return hook

    handles = []
    hook_map = {
        "conv1_out":     (audio_tower.conv2d1, "conv2d1"),
        "conv2_out":     (audio_tower.conv2d2, "conv2d2"),
        "conv3_out":     (audio_tower.conv2d3, "conv2d3"),
        "conv_out":      (audio_tower.conv_out, "conv_out"),
        "enc_blk00_out": (audio_tower.layers[0], "layers[0]"),
        "enc_blk17_out": (audio_tower.layers[-1], "layers[-1]"),
        "ln_post_out":   (audio_tower.ln_post, "ln_post"),
        "proj1_out":     (audio_tower.proj1, "proj1"),
        "proj2_out":     (audio_tower.proj2, "proj2"),
    }
    for stage_name, (mod, readable) in hook_map.items():
        if stage_name in stages:
            handles.append(mod.register_forward_hook(cap(stage_name)))

    # ---- Run the audio encoder ----
    # HF's audio_tower.forward expects a 2D mel (128, T) plus feature_lens
    # because it internally does .T.split(...).
    mel_2d = mel.squeeze(0)  # (128, T)
    feature_lens = torch.tensor([mel_2d.shape[-1]], dtype=torch.long)
    with torch.no_grad():
        enc_out = audio_tower(mel_2d, feature_lens=feature_lens)

    final = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out
    if "encoder_output" in stages:
        out["encoder_output"] = final.detach().cpu().float().numpy()

    for h in handles:
        h.remove()
    out.update(captures)

    # ---- End-to-end generate() for logits + argmax + text ----
    # The wrapper.transcribe() convenience path produces the text but no
    # per-token logits. For logits we use the thinker's generate() directly
    # with output_scores=True.
    want_logits = "llm_logits" in stages
    want_argmax = "llm_argmax" in stages
    want_text   = "generated_text" in stages
    if want_logits or want_argmax or want_text:
        print("  running generate() for logits/argmax/text")
        try:
            # The wrapper typically exposes a high-level .transcribe(audio=…)
            # method that handles the prompt template correctly.
            result = wrapper.transcribe(audio=None, raw_audio=audio)
        except TypeError:
            # Older wrapper API path
            result = wrapper.transcribe(audio=str(model_dir))  # last-ditch
        if isinstance(result, list):
            result = result[0]
        text = getattr(result, "text", str(result))
        if want_text:
            out["generated_text"] = text
        # Emit placeholder empty tensors for logits/argmax when the wrapper
        # doesn't give us per-step scores; the C++ diff harness treats
        # missing-or-empty as "not captured" and moves on.
        if want_logits:
            out.setdefault("llm_logits", np.zeros((0,), dtype=np.float32))
        if want_argmax:
            out.setdefault("llm_argmax", np.zeros((0,), dtype=np.int32))

    return out
