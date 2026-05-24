**Title:** `metal/ggml-alloc : two specific ggml_set_output calls on a long F32 graph cause Metal to write all-NaN`

---

This is a bug report rather than a patch. Two specific
`ggml_set_output` calls on intermediate tensors in a 14-block
diffusion UNet (the chatterbox-tts S3Gen conditional flow-matching
decoder) cause Apple Silicon Metal to produce all-NaN output for
the entire graph. Without those marks the output is finite
(degraded from a CPU reference by `cos_min ~0.94`, but no NaN).
Either mark alone is also finite. The bug requires the
combination.

CPU backend produces correct output in every configuration tested.

## Minimum repro

UNet1D denoiser graph: ~14 sub-blocks (1 down + 12 mid + 1 up +
final), ~396 mul_mats per pass, F32 activations throughout. The
two trigger tensors:

1. `dump_db_resnet` — output of the first DOWN block's
   `causal_resnet_block` (shape `(T_mel, 256)` F32, here T_mel=102).
2. Any one of `dump_mb_{0, 2, 4, 6, 8, 10, 11}_out` — output of an
   *even-indexed* MID block, or the last one (`mb_11`). Same shape.

Marking both with `ggml_set_output` → Metal writes all-NaN to the
final output. Either alone → output is finite. Marking an
odd-indexed `mb_*_out` (1, 3, 5, 7, 9) instead → finite,
deterministic.

## Backstory — why `set_output` was added

`ggml_set_output` was added as a debug aid: without it, `ggml_gallocr`
reuses the buffer of any tensor satisfying
`hn->n_children == 1 && hn->n_views == 0` for downstream ops
(see `ggml-alloc.c` around line 644). The debug dumper walked the
graph and read every node whose name started with `dump_*` — but
because some weren't `set_output`'d, their buffer had been
recycled by the time `ggml_backend_tensor_get` ran, and the dump
was stale data.

Adding `ggml_set_output` to each dump point made the dump
reliable. It also revealed the NaN.

## Why this is reportable upstream

- Same model + same graph + same inputs + identical seed.
- Only difference between "Metal output finite" and "Metal output
  all-NaN" is two specific `ggml_set_output` calls.
- CPU backend: every configuration finite and correct.

The parity pattern (even-indexed mb_*_out marks trigger; odd
don't) is geometric, not aliasing. A per-node `ggml_gallocr` trace
of both configurations (instrumentation merged downstream) shows
that even-indexed `mb_*_out` tensors land at `chunk=0 offset=0`
and odd-indexed at `chunk=0 offset=1271296` in the un-marked
control. Pinning the low-offset slot via `set_output` →
`FREE_SKIP_OUTPUT` blocks reuse of offset=0 and forces ~1300
downstream allocations to shifted positions. Pinning the
mid-offset slot only creates a small hole and a minor shift.

An overlap scan over the allocator trace of both 1-mark and 2-mark
configurations finds **zero overlapping live byte-ranges** in
either pass — the allocator's plan is correct in both cases. The
bug is therefore in the backend layer: a Metal kernel whose
correctness depends on the specific address pattern produced when
downstream allocations are shifted, or the output-staging path
that handles `set_output`-flagged tensors, or sched-level
interaction between OUTPUT tensors and split boundaries.

## What's been ruled out

These were tried during a long debug session and none of them is
the source:

1. **`mul_mat` kernel precision.** Added a `GGML_PREC_F32` Q8_0
   path that's bit-identical to CPU's
   `ggml_vec_dot_q8_0_q8_0_generic` (filed as separate PR 09).
   With this kernel firing on all 350 prec-tagged UNet mul_mats,
   the baseline drift moves from `cos_min 0.940` to `0.947`
   (essentially no change), and the 2-mark NaN trigger is
   unchanged.
2. **Per-op pin to CPU.** Pinning *any* frequent op (norm, mul,
   add, flash_attn_ext, gelu, reshape, cont, concat, permute,
   mul_mat) restores `cos_min = 1.000` on the baseline. Pinning a
   sparse op (conv_1d, soft_max, mish, silu, scale) doesn't help.
   This is a sync-barrier-density signal, not an op-identity
   signal. Doesn't address the 2-mark NaN.
3. **`GGML_NO_INPLACE=1`.** A global allocator knob to skip
   in-place reuse → `cos_min = -0.97` (sign-flipped garbage).
   Some downstream code (we didn't trace which) depends on
   in-place semantics.
4. **`GGML_METAL_CONCURRENCY_DISABLE=1`.** No effect on either
   the baseline drift or the 2-mark NaN. Whatever the bug is, it
   isn't a missing barrier between *parallel* command buffers.
5. **`kernel_norm` F32 audit.** `kernel_norm_fuse_impl` uses
   `float sumf` accumulators end-to-end, F32 `simd_shuffle_xor`
   reduction, F32 shmem. Clean.
6. **`kernel_flash_attn_ext` Q downcast.** Line 6430 downcasts Q
   from F32 to half: `sq4[j*DK4 + i] = (q4_t) q4[i]` where
   `q4_t = half4` even in the `FA_TYPES_F32` family. Patched to
   `float4 / simdgroup_float8x8`; doesn't fix the 2-mark NaN
   (and changes baseline drift in unhelpful ways).

## Investigation pointers

- `ggml-alloc.c:622` `ggml_gallocr_allocate_node` — *not* the
  source (verified by per-node trace + overlap scan). Adding the
  2-mark `set_output` shifts ~1300 downstream allocation offsets
  but produces no live-range overlap; the plan is valid.
- `ggml-metal/ggml-metal-ops.cpp:159`
  `ggml_metal_op_concurrency_check` — barrier insertion against
  the shifted address layout. Disabling concurrency entirely had
  no effect on the 2-mark NaN; this code path is unlikely to be
  the direct cause.
- The Metal output-staging path that copies `set_output`-flagged
  tensors out of device memory — concurrent staging copies
  alongside in-flight kernels writing to neighbouring tensors are
  a plausible candidate, especially when downstream kernels are
  now reading from offsets that wouldn't normally co-occur with
  the pinned output.
- Backend-sched interaction with `GGML_TENSOR_FLAG_OUTPUT` at
  split boundaries.

## How to reproduce

Standalone repro needs the chatterbox model files (50 MB
Q8_0 chatterbox-s3gen + Q8_0 chatterbox-t3) and a reference WAV.
We can extract a minimal `test-backend-ops`-style case from this
graph — flag this issue and we'll prepare one.

Application-level repro (with our project's source):

```
# baseline (1 mark or 0 marks) — finite
CRISPASR_CHATTERBOX_FORCE_GPU=1 \
CRISPASR_S3GEN_UNET_GPU_RESIDENCY=1 \
CRISPASR_S3GEN_UNET_MARK_DB_RESNET=1 \
  ./crispasr --backend chatterbox --tts "Hello." \
  --voice samples/jfk.wav --tts-output /tmp/cb.wav --seed 42

# 2-mark trigger (DB_RESNET + mb_0_out, even index) — NaN
CRISPASR_CHATTERBOX_FORCE_GPU=1 \
CRISPASR_S3GEN_UNET_GPU_RESIDENCY=1 \
CRISPASR_S3GEN_UNET_MARK_DB_RESNET=1 \
CRISPASR_S3GEN_UNET_MARK_MB_OUT_INDEX=0 \
  ./crispasr --backend chatterbox --tts "Hello." \
  --voice samples/jfk.wav --tts-output /tmp/cb.wav --seed 42

# 2 marks with odd index — finite
CRISPASR_S3GEN_UNET_MARK_MB_OUT_INDEX=1 \
[...rest same as above...]
```

## Production workaround

We split the UNet weight residency: the 910 `s3.fd.*` tensors load
on the CPU backend; the surrounding encoder/vocoder stay on GPU.
The scheduler routes the UNet sub-graph to CPU based on weight
residency. M1 wall-time comparable to pure CPU. Eliminates the
2-mark NaN (and the baseline drift) because the UNet no longer
runs on Metal at all. Not a fix — just a way to ship intelligible
audio while the kernel-side bug is open.
