# VoxCPM2 TTS: Debug numerical correctness via crispasr-diff

Continue debugging VoxCPM2 TTS in CrispASR (whisper.cpp fork at /mnt/akademie_storage/whisper.cpp).

## Problem

The pipeline runs end-to-end but produces **noise instead of speech**. Tokenization is verified correct (matches PyTorch). The issue is in the transformer forward passes — likely LocDiT or the CFM Euler solver.

## What's done and working

1. **src/voxcpm2_tts.cpp** (2461 lines) — full inference engine:
   - mmap loading via `core_gguf::load_weights` (580 MB RSS for Q4_K)
   - ggml_mul_mat for all matmuls (5-6× speedup)
   - SentencePiece BPE tokenizer (verified correct)
   - TSLM 28L prefill + step with KV cache
   - RALM 8L prefill + step
   - LocEnc 12L bidirectional + CLS token
   - LocDiT 12L + CFM Euler solver
   - VAE decoder (causal transposed conv + Snake1d + weight-norm)
   - Timing instrumentation (prefill, AR step, VAE)

2. **Models on disk:**
   - `/mnt/storage/voxcpm2-f16.gguf` — 4.63 GB F16
   - `/mnt/storage/voxcpm2-q4_k.gguf` — 1.6 GB Q4_K
   - `/mnt/akademie_storage/voxcpm2_hf/` — HF checkpoint (safetensors + audiovae.pth + tokenizer.json)

3. **PyTorch source:** `/tmp/voxcpm_src/voxcpm/` (pip install --target)

4. **Build:** `cd /mnt/akademie_storage/whisper.cpp/build-test && make -j4 crispasr-cli`

5. **Test command:**
   ```bash
   cd /mnt/akademie_storage/whisper.cpp/build-test
   VOXCPM2_INFERENCE_STEPS=1 VOXCPM2_MAX_LEN=3 \
     ./bin/crispasr --backend voxcpm2-tts \
       -m /mnt/storage/voxcpm2-q4_k.gguf -t 4 \
       --tts "Hi" --tts-output /tmp/test.wav
   ```

6. **HuggingFace:** `cstr/voxcpm2-GGUF` (F16 + Q4_K uploaded)

## What's broken

The generated audio is noise. ASR (parakeet) detects English but produces no transcript. Root cause: transformer outputs are numerically wrong somewhere in the pipeline.

## Debugging approach: crispasr-diff

### Step 1: Fix the PyTorch reference dumper

File: `tools/reference_backends/voxcpm2_tts.py`

The reference backend loads the PyTorch model and captures stage-by-stage activations. It currently crashes at:
```
model.base_lm.kv_cache.reset()
AttributeError: 'StaticKVCache' object has no attribute 'reset'
```

The VoxCPM2 model's KV cache API is `model.base_lm.kv_cache` which is a `StaticKVCache`. Check what methods it has:
```python
# In /tmp/voxcpm_src/voxcpm/modules/minicpm4/cache.py
class StaticKVCache:
    def fill_caches(self, kv_cache_tuple): ...
    def step(self): ...
    # May need: .clear() or just re-init
```

Fix: replace `model.base_lm.kv_cache.reset()` with proper cache reset (probably `model.base_lm.setup_cache(1, max_len, device, dtype)` or just skip reset if running only one inference).

### Step 2: Capture key stages

The reference backend should capture these activations (defined in DEFAULT_STAGES):
- `text_input_ids` — token IDs after BPE + CJK expansion (verify C++ matches)
- `tslm_prefill_out` — TSLM output after FSQ masking [seq_len, 2048]
- `locenc_out` — LocEnc CLS output for first AR step [1024]
- `lm_to_dit_hidden` — projected LM hidden for DiT [1024]
- `dit_step0_input` — LocDiT input sequence (mu + t_embed + cond + x)
- `cfm_step0_result` — CFM output after Euler steps [64, 4]
- `decoded_audio` — first 1s of VAE-decoded audio

### Step 3: Run C++ with stage extraction

The C++ `voxcpm2_extract_stage()` function is implemented but may need the same stage names. Use:
```bash
./bin/crispasr-diff --backend voxcpm2-tts \
  --model /mnt/storage/voxcpm2-f16.gguf \
  --reference /tmp/voxcpm2-ref.gguf
```

### Step 4: Compare stage by stage

Expected: cos_min >= 0.999 for transformer outputs. Any stage that drops below 0.99 indicates a bug in the C++ implementation of that component.

## Key architecture details for debugging

### TSLM forward (per-token step)
```
input [2048] → RMSNorm → Q/K/V proj → LongRoPE → GQA (16h, 2kv) → O proj → residual
            → RMSNorm → SwiGLU FFN (gate×silu, up, down) → residual
```
LongRoPE: uses `tslm.rope_short_factors` [64] when pos < 32768, else `tslm.rope_long_factors`.

### LocDiT forward (called per CFM step, double-batch for CFG)
```python
x = in_proj(x_raw.T)           # [N, P=4, 64] → [N, P, 1024]
cond = cond_proj(cond_raw.T)   # [N, P, 64] → [N, P, 1024]
t_emb = time_mlp(sinusoidal(t)) + dt_mlp(sinusoidal(dt=0))  # [N, 1024]
mu_reshaped = mu.view(N, 2, 1024)   # mu is [N, 2048]
seq = cat[mu_reshaped, t_emb.unsqueeze(1), cond, x]  # [N, 2+1+P+P=11, 1024]
hidden = 12L_bidirectional_transformer(seq)
out = out_proj(hidden[:, 2+1+P:, :])  # take last P tokens → [N, P, 64]
return out.transpose(1,2)  # [N, 64, P]
```

### CFM Euler solve
```python
z = randn(1, 64, P=4) * temperature
t_span = linspace(1, 0, N+1) + sway*(cos(pi/2*t) - 1 + t)  # sway=1.0
for step 1..N:
    if step <= zero_init_steps(=1): dphi_dt = 0
    else:
        x_in = [x, x]; mu_in = [mu, zeros]; t_in = [t, t]
        cond_in = [cond, cond]; dt_in = zeros
        est = LocDiT(x_in, mu_in, t_in, cond_in, dt_in)
        pos, neg = split(est)
        st_star = dot(pos,neg) / (||neg||^2 + 1e-8)
        dphi_dt = neg*st_star + cfg*(pos - neg*st_star)
    x = x - dt * dphi_dt
```

### Stop predictor
```
h = stop.proj(lm_hidden) + bias    # [2048] → [2048]
h = silu(h)
logits = stop.head(h)              # [2048] → [2], no bias
stop = argmax(logits) == 1
```

### Fusion before RALM
```
ralm_input = fusion.weight @ cat(fsq_output, enc_to_lm_proj(locenc_out)) + fusion.bias
# Input: [4096] = [2048 + 2048], Output: [2048]
```

## Likely bug locations (in order of probability)

1. **LocDiT sequence construction** — the `mu.view(N, 2, 1024)` reshape, the order of concatenation `[mu, t, cond, x]`, or the output slicing `hidden[:, prefix+2+1:, :]`

2. **CFM dt calculation** — the `t_span[step] - t_span[step+1]` step size may be wrong, or the sway coefficient is applied incorrectly

3. **RoPE in TSLM** — LongRoPE factors may be applied wrong (the factors are [64] for head_dim/2=64 rotary dimensions)

4. **Attention scale** — should be `1/sqrt(head_dim)` = `1/sqrt(128)` ≈ 0.0884

5. **KV cache position tracking** — off-by-one in position IDs passed to RoPE

## Python dependencies needed
```bash
pip install --break-system-packages einops pydantic librosa torchaudio
# Already installed: torch, transformers, numpy, safetensors
```

## File map
```
src/voxcpm2_tts.cpp                    — C++ inference (2461 lines)
src/voxcpm2_tts.h                      — public C API
examples/cli/crispasr_backend_voxcpm2_tts.cpp — CLI adapter
tools/reference_backends/voxcpm2_tts.py — PyTorch reference dumper
tools/dump_reference.py                — unified reference dump CLI
models/convert-voxcpm2-to-gguf.py      — HF→GGUF converter
/tmp/voxcpm_src/voxcpm/model/voxcpm2.py — PyTorch model source
/tmp/voxcpm_src/voxcpm/modules/locdit/  — LocDiT + UnifiedCFM source
/tmp/voxcpm_src/voxcpm/modules/locenc/  — LocEnc source
/tmp/voxcpm_src/voxcpm/modules/minicpm4/ — MiniCPM-4 transformer source
```

## Benchmarks (current, Q4_K, 4 threads)
| Phase | Time |
|-------|------|
| Prefill (2 tokens) | 7s |
| AR step (1 CFM) | 1.5s |
| VAE decode (3 patches) | 26s |

## Memory
- Q4_K: 580 MB RSS (mmap)
- F16: 3.7 GB RSS (page cache)
- Machine: 7.6 GB RAM total

## Strategy

1. Fix the `StaticKVCache.reset()` issue in the Python reference backend
2. Run `dump_reference.py` with just `text_input_ids` + `tslm_prefill_out` stages (minimal memory)
3. Compare C++ TSLM output against reference — if cos < 0.99, the bug is in TSLM
4. If TSLM matches, move to LocDiT: capture `dit_step0_input` and `cfm_step0_result`
5. Fix whatever stage diverges
6. Once cos >= 0.999 for all stages, the audio should be recognizable
7. Run ASR roundtrip to confirm

## Quick validation (no PyTorch needed)

Even without the reference dump, you can validate basic numerical sanity:
- TSLM output should have reasonable magnitude (not all zeros, not exploding)
- FSQ output should be in [-1, 1] range (tanh + round)
- CFM output (pred_feat) should be in reasonable range for audio latents
- VAE output should be in [-1, 1] (tanh at output)

Add fprintf debugging in `voxcpm2_synthesize_internal` to print:
- min/max/mean of tslm_hidden after prefill
- min/max of fsq_out
- min/max of cfm result per AR step
- min/max of VAE output
