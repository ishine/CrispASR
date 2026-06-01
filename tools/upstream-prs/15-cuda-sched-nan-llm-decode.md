**Title:** `ggml-backend / sched : dual-backend [CUDA,CPU] produces Inf/NaN in LLM decoder at layer 2+ (funasr Qwen2-0.6B)`

---

Same bug class as issue #11 (Metal sched NaN at large T), now confirmed
on CUDA. The `ggml_backend_sched` with `[CUDA, CPU]` backends misroutes
LLM decoder ops, producing Inf at layer 2 and all-NaN by layer 3.

## Symptom

A 28-layer Qwen2-0.6B LLM decoder (head_dim=128, n_heads=16,
n_kv_heads=8, GQA ratio=2, per-head QK-RMSNorm) runs correctly on
CPU-only builds but produces all-NaN prefill logits on CUDA when
scheduled through `ggml_backend_sched_new({cuda, cpu}, ..., false, false)`.

Per-layer tensor dump:
```
llm_layer_0:  0 NaN, 0 Inf  — matches CPU
llm_layer_1:  0 NaN, 0 Inf  — matches CPU
llm_layer_2:  0 NaN, 1 Inf  — max=6973 vs CPU max=124512
llm_layer_3:  ALL NaN (47104/47104)
llm_layer_4+: ALL NaN
```

The encoder (70-layer SANM, same sched, same graph compute call) is
fine — 0 NaN across all layers, values match CPU to <0.05 first8 diff.

## Confirmed on

- Tesla P100 (sm_60) — Kaggle, 16 kernel versions
- Blackwell (sm_120) — GitHub issue #125

Architecture-independent within CUDA.

## What does NOT fix it

| Attempt | Result |
|---|---|
| Q8_0 weights (instead of F16) | Still NaN |
| Disable flash_attn_ext (`FUNASR_NO_FA=1`) | Still NaN |
| F32 KV cache reads (`CRISPASR_KV_READ_F32=1`) | Still NaN |
| `parallel=true` sched flag | Still NaN |
| Fuse Q/K/V into single QKV matmul | Still NaN |
| Single-backend GPU sched (remove CPU) | Crashes (ops need CPU) |
| Zero-fill KV cache on alloc | Still NaN |

## What DOES fix it

Put the LLM decoder's weight tensors + KV cache on the CPU backend
(via `load_weights_split`). The sched then routes the entire LLM
subgraph to CPU. Encoder stays GPU-accelerated. Identical workaround
to issue #11 (Metal NaN at large T).

## Repro

Model: `cstr/funasr-nano-GGUF` → `funasr-nano-2512-q8_0.gguf` (HuggingFace).
Audio: any 16 kHz WAV (11s JFK speech used for testing).

```bash
# Build with CUDA
cmake -B build -DGGML_CUDA=ON && cmake --build build --target crispasr-cli

# CUDA: produces "!!!!!!!!!!!!!!!!!!!!" (all-NaN logits)
FUNASR_DUMP_STAGES=1 build/bin/crispasr --backend funasr -m auto \
    --auto-download -f samples/jfk.wav --no-prints

# CPU: produces correct transcript
CUDA_VISIBLE_DEVICES="" build/bin/crispasr --backend funasr -m auto \
    --auto-download -f samples/jfk.wav --no-prints
```

The `FUNASR_DUMP_STAGES=1` env var prints per-layer tensor stats to
stderr — look for NaN/Inf counts in `llm_layer_2` and beyond.

## Relation to existing upstream PRs

- **#10 (dangling src[j] pointers):** Independent bug, already patched
  in our fork. The funasr NaN persists with #10's fix applied.
- **#11 (Metal sched NaN at large T):** Same bug class. Different
  backend (CUDA vs Metal), different symptom shape (layer-2 Inf vs
  all-NaN-from-start), but same root cause (mixed-backend scheduling
  produces corrupted intermediate tensors). Same workaround (force
  entire subgraph to one backend via weight placement).
- **#12 (parallel=true shared-storage docs):** Not the same issue.
  `parallel=true` does NOT fix this bug (tested in Kaggle v8).

## Proposed investigation

The sched assigns each graph node to a backend based on weight
residency + op support. With all weights on GPU, the sched should
route everything to CUDA. But something at layer 2 gets misrouted or
the cross-backend data copy is corrupted. A `GGML_SCHED_DEBUG=2`
trace showing the per-node backend assignment + the split boundaries
for this graph would identify the exact misrouted op.

We can provide:
- The GGUF model file (1.5 GB Q8_0)
- A minimal C repro that builds and runs the LLM graph
- `GGML_SCHED_DEBUG=2` traces for the working (CPU) and broken (CUDA) runs

---

**Application-side workaround (shipped in our fork, commit `f94fec90`):**
`load_weights_split` + KV-on-CPU for the LLM decoder. The encoder
runs on CUDA for the GPU speedup. Performance hit: ~4.5× slower LLM
decode (CPU vs GPU), acceptable for ASR (short output sequences).
