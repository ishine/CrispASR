#!/usr/bin/env python3
"""
Convert Meta AudioSeal PyTorch checkpoints to GGUF.

Downloads generator_base.pth and detector_base.pth from
facebook/audioseal on HuggingFace, fuses weight_norm
reparametrization, and emits a single combined GGUF.

Architecture (SEANet encoder-decoder):
  - Encoder: Conv1d(1,32,7) → 4 blocks (ratios [2,4,5,8]) → LSTM(512,2) → Conv1d(512,128,7)
  - Message: Embedding(32,128) + Linear(128,128)
  - Decoder: mirrors encoder with ConvTranspose1d upsampling
  - Detector: encoder + ConvTranspose1d(128,32,320) + Conv1d(32,18,1)

Weight naming in GGUF:
  audioseal.gen.enc.in.weight       — Conv1d(1, 32, k=7)
  audioseal.gen.enc.blk.{i}.res.{j}.conv{k}.weight
  audioseal.gen.enc.blk.{i}.down.weight
  audioseal.gen.lstm.{i}.weight_ih  — LSTM layer i
  audioseal.gen.enc.out.weight      — Conv1d(512, 128, k=7)
  audioseal.gen.msg.embedding       — Embedding(32, 128)
  audioseal.gen.msg.linear.weight   — Linear(128, 128)
  audioseal.gen.dec.*               — decoder (mirrors encoder)
  audioseal.det.*                   — detector

Usage:
    pip install audioseal gguf
    python models/convert-audioseal-to-gguf.py --output audioseal.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


def fuse_weight_norm(state_dict: dict, prefix: str) -> dict:
    """Fuse weight_norm g/v pairs into a single weight tensor."""
    fused = {}
    processed = set()

    for key in list(state_dict.keys()):
        if key.endswith(".parametrizations.weight.original0"):
            base = key.rsplit(".parametrizations.weight.original0", 1)[0]
            g_key = key
            v_key = base + ".parametrizations.weight.original1"
            if v_key in state_dict:
                g = state_dict[g_key]  # (C_out, 1, 1) or (C_out,)
                v = state_dict[v_key]  # (C_out, C_in, K) or similar
                # Fuse: w = g * v / ||v||
                v_norm = torch.linalg.norm(v.reshape(v.shape[0], -1), dim=1)
                while v_norm.dim() < v.dim():
                    v_norm = v_norm.unsqueeze(-1)
                g = g.reshape(v.shape[0], *([1] * (v.dim() - 1)))
                w = g * v / (v_norm + 1e-12)
                out_key = prefix + base + ".weight"
                fused[out_key] = w
                processed.add(g_key)
                processed.add(v_key)
                continue

        if key not in processed:
            out_key = prefix + key
            fused[out_key] = state_dict[key]

    return fused


def remap_keys(tensors: dict, prefix: str) -> dict:
    """Remap PyTorch state_dict keys to GGUF naming convention."""
    remapped = {}
    for key, val in tensors.items():
        # Strip the prefix added during fuse_weight_norm
        k = key
        if k.startswith(prefix):
            k = k[len(prefix):]

        # Map SEANet encoder/decoder layers to our naming
        # PyTorch: encoder.model.0.weight → audioseal.{gen|det}.enc.in.weight
        # PyTorch: encoder.model.{2i+1}.block.{j}... → audioseal.{gen|det}.enc.blk.{i}.res.{j}...
        # etc.

        # For now, store with a cleaned-up key using the original structure
        # The C++ bind_tensors will match these names
        out_key = "audioseal." + k.replace("model.", "").replace("block.", "res.")
        remapped[out_key] = val.float().numpy()

    return remapped


def main():
    ap = argparse.ArgumentParser(description="Convert AudioSeal to GGUF")
    ap.add_argument("--output", "-o", required=True, help="Output GGUF path")
    ap.add_argument("--generator", help="Path to generator_base.pth (or auto-download)")
    ap.add_argument("--detector", help="Path to detector_base.pth (or auto-download)")
    args = ap.parse_args()

    # Try to load from paths or download
    gen_state = None
    det_state = None

    if args.generator:
        gen_state = torch.load(args.generator, map_location="cpu", weights_only=True)
    else:
        try:
            from huggingface_hub import hf_hub_download
            gen_path = hf_hub_download("facebook/audioseal", "generator_base.pth")
            gen_state = torch.load(gen_path, map_location="cpu", weights_only=True)
            print(f"Downloaded generator from facebook/audioseal")
        except Exception as e:
            print(f"Could not download generator: {e}", file=sys.stderr)

    if args.detector:
        det_state = torch.load(args.detector, map_location="cpu", weights_only=True)
    else:
        try:
            from huggingface_hub import hf_hub_download
            det_path = hf_hub_download("facebook/audioseal", "detector_base.pth")
            det_state = torch.load(det_path, map_location="cpu", weights_only=True)
            print(f"Downloaded detector from facebook/audioseal")
        except Exception as e:
            print(f"Could not download detector: {e}", file=sys.stderr)

    if gen_state is None and det_state is None:
        print("Error: no generator or detector checkpoint found", file=sys.stderr)
        sys.exit(1)

    # Fuse weight_norm and collect tensors
    all_tensors = {}
    if gen_state:
        fused = fuse_weight_norm(gen_state, "gen.")
        all_tensors.update(fused)
        print(f"Generator: {len(fused)} tensors")

    if det_state:
        fused = fuse_weight_norm(det_state, "det.")
        all_tensors.update(fused)
        print(f"Detector: {len(fused)} tensors")

    # Write GGUF
    writer = GGUFWriter(args.output, "audioseal")

    # Metadata
    writer.add_uint32("audioseal.sample_rate", 16000)
    writer.add_uint32("audioseal.dimension", 128)
    writer.add_uint32("audioseal.n_filters", 32)
    writer.add_uint32("audioseal.n_residual_layers", 1)
    writer.add_uint32("audioseal.nbits", 16)
    writer.add_uint32("audioseal.lstm_layers", 2)
    writer.add_array("audioseal.ratios", [8, 5, 4, 2])

    # Tensors
    n_written = 0
    for name, data in sorted(all_tensors.items()):
        arr = data if isinstance(data, np.ndarray) else data.numpy()
        arr = arr.astype(np.float32) if arr.dtype != np.float32 else arr

        # Use F16 for large weight tensors, F32 for biases/small
        if arr.size > 256 and "bias" not in name and "alpha" not in name:
            arr_f16 = arr.astype(np.float16)
            writer.add_tensor(name, arr_f16, raw_dtype=GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote {n_written} tensors to {args.output}")
    total_bytes = Path(args.output).stat().st_size
    print(f"File size: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
