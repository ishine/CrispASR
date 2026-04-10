> **STATUS (2026-04-10): HISTORICAL.** These optimization plans were implemented.
> See TODO.md for current performance work.

# Closing the gap to ONNX INT8 in CrispASR

## Current state (5.4 s clip, 8 threads, Cascade Lake-class CPU)

| Path | Inference | Total wall |
| --- | ---: | ---: |
| ggml Q4_K (`cohere-main`) | ~15 s | ~15 s |
| ggml Q8_0                 | ~24 s | ~24 s |
| ggml F16                  | ~37 s | ~37 s (with `COHERE_MKL=ON`) |
| ONNX INT4 (`MatMulNBits`) | 9.7 s | 17.2 s (incl. 7.5 s cold load) |
| ONNX INT8 (`MatMulNBits`, bs=32) | 9.4 s | 27.0 s (incl. 17.5 s cold load) |
| ONNX INT8 (Tristan)       | 9.8 s | 36.3 s (incl. 26.5 s cold load) |

**ggml is already winning end-to-end** (no cold load thanks to mmap), but it is **5-6 s behind on pure inference**. The entire gap is in the matmul kernels: ggml's `vec_dot_q4_K_q8_K` uses AVX2 `pmaddubsw`/`pmaddwd`, while ONNX MLAS uses `vpdpbusd` (AVX-VNNI / AVX512-VNNI), which does a fused 4-way `u8 × s8 → s32` dot product per cycle.

## Why `COHERE_MKL=ON` doesn't fix this

`GGML_BLAS=ON` only routes **F32 GEMM** through MKL. For quantised types ggml dequantises a block to F32 and then calls `cblas_sgemm`. That helps F16 (+15 % measured) and barely moves Q5_0/Q8_0/Q4_K. Closing the gap properly requires calling MKL/oneDNN's **int8 GEMM** entry points (`cblas_gemm_s8u8s32` or `dnnl::matmul` with int8 src/weight descriptors), with quantised activations, not F32. ggml's BLAS hook doesn't expose those.

## Three options, in order of effort vs payoff

### Option 1 — Native VNNI Q8_0 kernel in ggml-cpu (1-2 weeks)

**Plan.** Add an x86 VNNI dispatch for `ggml_vec_dot_q8_0_q8_0`. Detect AVX-VNNI / AVX512-VNNI at runtime; fall back to the existing AVX2 kernel on older silicon.

**Files to touch.**
- `ggml/src/ggml-cpu/arch/x86/quants.c` — add `ggml_vec_dot_q8_0_q8_0_avxvnni` and `_avx512vnni` variants. Template: the existing `Q4_0_8_8` VNNI kernel which already uses `vpdpbusd`. Drop the unpack step since Q8 weights are already int8.
- `ggml/src/ggml-cpu/ggml-cpu.c` — wire the new kernels into the dispatch table in `ggml_cpu_init`.
- `ggml/src/ggml-cpu/cpu-feats-x86.cpp` — already detects VNNI; no changes.

**Expected impact.** Q8_0 wall time drops from ~24 s to ~12-14 s on Cascade Lake. That puts ggml Q8_0 ahead of ONNX INT8 end-to-end (no cold load). Q4_0 inherits the speedup automatically because its dot-product step quantises activations to Q8 first; only the Q4_0 unpack remains as overhead.

**Q4_K is harder** and probably not worth touching: its 6-bit super-block layout means the dot product can't use plain `vpdpbusd` directly; you would need a fused dequant+VNNI kernel and the existing AVX2 implementation is already close to silicon limits for the Q4_K format.

**Why option 1 first.** No new dependencies, one new file, the win lands across the entire ggml ecosystem (whisper.cpp, llama.cpp, cohere.cpp) simultaneously, and the ggml maintainers will likely accept it upstream.

### Option 2 — `ggml-onednn` backend (3-4 weeks)

**Plan.** New backend module `ggml/src/ggml-onednn/`, parallel to `ggml-blas`, `ggml-cuda`, `ggml-metal`. Calls `dnnl::matmul` with int8 src/weight descriptors. Caches primitives per `(M, N, K)` shape — primitive creation is ~ms, reuse is ~µs.

**Upsides.**
- Free AMX (Sapphire Rapids) and AVX10 support, oneDNN handles dispatch.
- Free Intel GPU backend via `dnnl::engine::kind::gpu`.
- Inherits oneDNN's tuning across CPU generations.

**Downsides.**
- ~30 MB oneDNN runtime dependency (similar to MKL).
- More code (~2500 LOC), harder to debug, slower convergence.
- Activation quantisation logic has to be plumbed through `ggml_compute_forward_mul_mat`, which is invasive.

**Expected impact.** ~50 % speedup over option 1 on Sapphire Rapids+ (where AMX dominates). On older x86 the gap to option 1 is small.

### Option 3 — Stay with Q4_K, document the gap

The honest baseline. Q4_K is the right default for the README's audience: laptop, M-series Mac, ARM server, hobbyist running on CPU. The 5-second gap to ONNX INT8 only exists on x86 Cascade Lake+ servers, and that workload is mostly migrating to GPU anyway. ggml wins on cold load, memory, portability, and ARM.

## Recommendation

**Option 1.** Concretely:

1. Build a benchmark harness in `tests/` that measures `ggml_vec_dot_q8_0_q8_0` throughput in isolation (token/s, not end-to-end), so we can iterate on the kernel without re-running the full Cohere pipeline.
2. Copy `q4_0_8_8` VNNI kernel as the starting template, strip the unpack step.
3. Verify correctness against the existing AVX2 path on a small unit test.
4. Plumb into the dispatch table.
5. Re-run the Cohere benchmark to confirm Q8_0 inference drops to the 12-14 s range.
6. PR to ggml upstream.

This is a few-day project for someone familiar with the ggml-cpu codebase, and the win lands across whisper.cpp / llama.cpp / cohere.cpp simultaneously. Skip options 2 and 3 unless option 1 turns out to be insufficient.

## Where this does *not* matter

- **Apple Silicon.** ggml's NEON Q4_K already wins by a wide margin. MLAS doesn't have the same NEON tuning. Don't bother.
- **ARM in general.** Same story.
- **CUDA / Metal / SYCL.** Already covered by the existing ggml backends. None of this is GPU work.
- **Whisper-style models.** whisper.cpp Q5_0 already runs at ~1× realtime on most CPUs; the bottleneck is decode latency, not GEMM throughput.
