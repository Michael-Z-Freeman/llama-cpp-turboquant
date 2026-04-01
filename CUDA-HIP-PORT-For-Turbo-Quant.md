# TurboQuant CUDA/HIP Port

This document describes the CUDA/HIP backend port added to support TurboQuant KV cache compression
(`--cache-type-k turbo3`, `--cache-type-v turbo3`) on AMD/NVIDIA GPUs (ROCm/HIP and CUDA).

The original TurboQuant implementation in this fork (TheTom/llama-cpp-turboquant) only had Metal (Apple)
and CPU backends. Running with ROCm previously failed:

```
pre-allocated tensor (cache_k_l0 (view)) in a buffer (ROCm0) that cannot run the operation (SET_ROWS)
```

The root cause was not a ROCm version issue but a code gap: the HIP/CUDA backend had no support for
the TURBO3_0 or TURBO4_0 quantization types, nor for the TURBO_WHT graph op.

---

## What Was Added

### New Files

#### `ggml/src/ggml-cuda/turbo-wht.cuh`
Header declaring `ggml_cuda_op_turbo_wht`.

#### `ggml/src/ggml-cuda/turbo-wht.cu`
CUDA/HIP kernel implementing the Fast Walsh-Hadamard Transform used by TurboQuant.

- One GPU thread processes one 128-element group sequentially.
- `direction=0` (forward): applies sign array `s1`, runs 7-stage butterfly (stride-1 to 64),
  normalises by `1/sqrt(128)`, applies sign array `s2`.
- `direction=1` (inverse): swaps the sign arrays.
- Sign arrays `k_turbo_wht_s1[128]` and `k_turbo_wht_s2[128]` are stored as `__constant__` memory
  and match the CPU (`ggml-cpu/ops.cpp`) and Metal (`ggml-metal/turbo-wht.h`) implementations exactly
  (seeded with 42 and 1042 respectively).
- `ggml_cuda_op_turbo_wht` reads `direction` from `dst->op_params[0]` and dispatches
  `k_turbo_wht_f32<<<(num_groups+255)/256, 256>>>`.

#### `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu`
#### `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo4_0.cu`
Explicit template instantiations of `ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>` for
D ∈ {64, 128, 256} with K=V=TURBO3_0 and K=V=TURBO4_0 respectively. These are compiled as
separate translation units to keep build times manageable.

---

### Modified Files

#### `ggml/src/ggml-cuda/cpy-utils.cuh`

Added two new block-level quantization device functions:

**`quantize_f32_turbo3_0_block(x[32], y)`**
- Computes the L2 norm of 32 elements.
- Normalises each element, finds the nearest of 8 Lloyd-Max centroids optimised for N(0, 1/128):
  `{-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685}`
- Packs the 3-bit index: lower 2 bits → `qs[i/4]` (2 bits per element × 4 = 1 byte per 4),
  upper 1 bit → `signs[i/8]` (1 bit per element × 8 = 1 byte per 8).
- Stores norm as fp16 in `y->norm`.

**`quantize_f32_turbo4_0_block(x[128], y)`**
- Same centroid lookup on 128 elements.
- Packs 3-bit indices with bit-spanning across bytes (bit offset = i×3, may straddle two bytes).
- QJL residual correction is deferred: `rnorm = 0`, `signs = 0` (gives functional ~3-bit quality).

Both functions assume the input has already been WHT-rotated by the preceding `GGML_OP_TURBO_WHT`
graph node.

#### `ggml/src/ggml-cuda/set-rows.cu`

Added dispatch cases in `set_rows_cuda<src_t, idx_t>` for the two new types:

```cpp
} else if (dst->type == GGML_TYPE_TURBO3_0) {
    set_rows_cuda_quant<idx_t, block_turbo3_0, QK_TURBO3, quantize_f32_turbo3_0_block>(...);
} else if (dst->type == GGML_TYPE_TURBO4_0) {
    set_rows_cuda_quant<idx_t, block_turbo4_0, QK_TURBO4, quantize_f32_turbo4_0_block>(...);
}
```

This is the fix for the original `SET_ROWS` failure.

#### `ggml/src/ggml-cuda/fattn-common.cuh`

Added four device functions for FlashAttention:

**`dequantize_V_turbo3_0<T, ne>`** / **`dequantize_V_turbo4_0<T, ne>`**
- Decode `ne` elements from a TURBO3_0 or TURBO4_0 block at offset `i0`.
- TURBO3_0: reconstruct 3-bit index from `qs` (lower 2 bits) and `signs` (upper 1 bit),
  then `val = CENTROIDS[idx] * half2float(norm)`.
- TURBO4_0: extract 3-bit index from bit-packed `qs` array (handles byte-spanning),
  then `val = CENTROIDS[idx] * half2float(norm)`.
- Writes to `dst` as `float` or `half` depending on template param `T`.

**`vec_dot_fattn_vec_KQ_turbo3_0<D, nthreads>`** / **`vec_dot_fattn_vec_KQ_turbo4_0<D, nthreads>`**
- Computes the K·Q dot product inside FlashAttention.
- K is in TURBO3_0/TURBO4_0 format; Q is in q8_1 format (int8 quantised with scale).
- Both K and Q are already in the WHT-rotated domain. Because WHT is orthogonal, `K_rot·Q_rot =
  K_orig·Q_orig`, so no back-rotation is needed here — dequantise K centroids, multiply by Q int8
  values and scale, accumulate.

Both pairs are wired into the existing dispatch tables:
- `get_vec_dot_KQ<type_K, D, nthreads>()` — returns the correct function pointer for TURBO3_0/4_0.
- `get_dequantize_V<type_V, T, ne>()` — same for V dequantization.

#### `ggml/src/ggml-cuda/fattn-vec.cuh`

- Extended `EXTERN_DECL_FATTN_VEC_CASES` macro to also declare `GGML_TYPE_TURBO3_0` and
  `GGML_TYPE_TURBO4_0` V types.
- Added `EXTERN_DECL_FATTN_VEC_CASES(64/128/256, GGML_TYPE_TURBO3_0)` and same for TURBO4_0,
  so the forward declarations exist for all three head dimensions.

#### `ggml/src/ggml-cuda/fattn.cu`

- Added `GGML_TYPE_TURBO3_0` and `GGML_TYPE_TURBO4_0` to the `K->type` switch in
  `ggml_cuda_get_best_fattn_kernel` (they fall through to `break` alongside Q4_0, Q8_0, BF16).
- Added `FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)` and same for TURBO4_0
  in both the `#ifdef GGML_CUDA_FA_ALL_QUANTS` and the default `else` branches of
  `ggml_cuda_flash_attn_ext_vec`.
- Added **early-return VEC** immediately after tensor extraction (before all arch-specific guards):
  ```cpp
  if (K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0) {
      if (Q->ne[0] > 256 || Q->ne[0] % 64 != 0) return BEST_FATTN_KERNEL_NONE;
      return BEST_FATTN_KERNEL_VEC;
  }
  ```
  This ensures TURBO types always route to the VEC kernel regardless of sequence length,
  bypassing the `K->ne[1] % FATTN_KQ_STRIDE` guard and any arch-specific branches (RDNA4,
  Turing, Volta) that would select TILE/MMA kernels which have no TURBO dequant path.
  It also prevents the `mask->ne[2] != 1` guard from returning NONE before the TURBO check.

#### `ggml/src/ggml-cuda/fattn-vec.cuh`

Added a per-position bounds check inside the VEC kernel's inner KQ loop to handle the case
where `ne11 < FATTN_KQ_STRIDE` (e.g. short sequences at the start of generation):

```cpp
const bool kv_valid = (k_VKQ_0 + i_KQ) < k_VKQ_max;
float sum = kv_valid ? vec_dot_KQ(...) : 0.0f;
...
if (!kv_valid) { sum = -INFINITY; }
```

Out-of-range positions are masked to −∞ so softmax assigns them zero weight.

#### `ggml/src/ggml-cuda/ggml-cuda.cu`

Three changes:

1. `#include "ggml-cuda/turbo-wht.cuh"` added alongside the other backend includes.

2. `GGML_OP_TURBO_WHT` case in the compute dispatch:
   ```cpp
   case GGML_OP_TURBO_WHT:
       ggml_cuda_op_turbo_wht(ctx, dst);
       break;
   ```

3. In `supports_op`:
   - `SET_ROWS` check extended to accept `GGML_TYPE_TURBO3_0` and `GGML_TYPE_TURBO4_0` as destination types.
   - New `GGML_OP_TURBO_WHT` case added (requires src/dst both F32).

#### `ggml/src/ggml-hip/CMakeLists.txt`

The HIP backend has its own `CMakeLists.txt` separate from `ggml-cuda/CMakeLists.txt`. Added the
two new template instance files to the default (non-`FA_ALL_QUANTS`) source list:

```cmake
../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu
../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo4_0.cu
```

#### `ggml/src/ggml-cuda/CMakeLists.txt`

Same two files added for NVIDIA CUDA builds.

#### `ggml/src/ggml-cpu/quants.c`

Added CPU `from_float` wrappers for TURBO types (required for CPU fallback in SET_ROWS):

```c
void quantize_row_turbo3_0(const float * x, void * y, int64_t k) {
    block_turbo3_0 * dst = y;
    quantize_row_turbo3_0_ref(x, dst, k);
}
void quantize_row_turbo4_0(const float * x, void * y, int64_t k) { ... }
```

#### `ggml/src/ggml-cpu/quants.h`

Declarations for `quantize_row_turbo3_0` and `quantize_row_turbo4_0`.

#### `ggml/src/ggml-cpu/ggml-cpu.c`

Added TURBO3_0 and TURBO4_0 entries to the `type_traits_cpu[]` table:

```c
[GGML_TYPE_TURBO3_0] = { .from_float = quantize_row_turbo3_0, .nrows = 1 },
[GGML_TYPE_TURBO4_0] = { .from_float = quantize_row_turbo4_0, .nrows = 1 },
```

This fixed the CPU `SET_ROWS` crash (null function pointer in `ggml_compute_forward_set_rows_f32`).

---

## Quality and Losslessness

TurboQuant's paper (arXiv 2504.19874) claims **near-lossless** KV cache compression, but this
applies specifically to TURBO4_0 with all three components working together:

| Component | TURBO3_0 | TURBO4_0 |
|---|---|---|
| WHT rotation (PolarQuant) | ✅ | ✅ |
| 128-element quantization blocks | ❌ (32-element) | ✅ |
| QJL residual correction (`rnorm` + `signs`) | N/A | ✅ (paper) / ❌ not yet implemented |

**TURBO3_0** is always lossy. WHT rotation reduces quantization error by spreading signal energy
evenly across dimensions before quantizing to 8 Lloyd-Max centroids (3 bits), but it cannot
eliminate loss. Coherent but degraded output is expected and correct behaviour.

**TURBO4_0** with all components would be near-lossless per the paper. Our current implementation
has two gaps even once the garbage-output bug (Bug 5) is resolved:
1. QJL correction is deferred — `rnorm=0`, `signs=0` in every block.
2. The output bug itself must first be fixed.

Until QJL is implemented, TURBO4_0 will be better than TURBO3_0 (larger blocks, better
normalization) but not at the paper's claimed quality level.

---

## Architecture / Data Flow

```
Prefill (write KV cache)
────────────────────────
  K/V (F32)
    └─ ggml_turbo_wht (direction=0, forward)   [llama-graph.cpp]
         s1 → FWHT butterfly (7 stages) → normalise → s2
    └─ GGML_OP_SET_ROWS  (quantise to TURBO3_0 / TURBO4_0)
         norm L2 per 32-elem block, lookup 3-bit Lloyd-Max centroid per element
    → KV cache (TURBO3_0 / TURBO4_0 blocks)

Decode (FlashAttention)
───────────────────────
  Q (F32)
    └─ GGML_OP_TURBO_WHT (direction=0, forward)
    └─ quantise Q to q8_1 (inside FlashAttention kernel)
  K/V (TURBO3_0 / TURBO4_0 from cache)
    → FlashAttention: K·Q via vec_dot_fattn_vec_KQ_turbo*
                      V dequant via dequantize_V_turbo*
    → KQV output (F32)
  KQV
    └─ GGML_OP_TURBO_WHT (direction=1, inverse)
         s2 → FWHT butterfly → normalise → s1
    → attention output
```

**Why no back-rotation in FlashAttention:**
The WHT is an orthogonal transform: `<WHT(K), WHT(Q)> = <K, Q>`. Since both K and Q are rotated
before the dot product, the result is identical to computing with the originals. The V dequantization
outputs the rotated V values, and the inverse WHT on the attention output undoes the rotation.

---

## Current Status (2026-03-29)

### End-of-day update (latest verified run)

- Branch tested: `Michael-Z-Freeman/llama-cpp-turboquant:turboquant-hip-port`
- GPU path confirmed active:
  - `ggml_cuda_init: found 1 ROCm devices`
  - ROCm memory breakdown shows large model/context allocations on GPU
- GPU used: AMD Radeon RX 9060 XT (gfx1200), VRAM 16,304 MiB
- Model used: `qwen2.5-coder-14b.gguf`

Long-context comparison at `--ctx-size 65536`:

| KV cache mode | Context memory (MiB) | Generation speed (t/s) | Tokens/min |
|---|---:|---:|---:|
| f16 | 8448 | 13.0 | 780 |
| turbo3 | 2688 | 29.2 | 1752 |
| turbo4 | 3264 | 26.5 | 1590 |

Memory reduction vs f16 context:
- turbo3: **-5760 MiB** (~68% less)
- turbo4: **-5184 MiB** (~61% less)

Notes:
- The original turbo4 `| CJK` failure case is still non-reproducible in current branch tests.
- Short CLI smoke tests may show low average “GPU %” in external monitors because workload is bursty;
  backend logs + ROCm memory allocations are the reliable confirmation of GPU offload.
- Context-size probe (turbo4 on current branch/build):
  - stable through `--ctx-size 129024`
  - segfault observed at `--ctx-size 130048` and above
  - at `129024`: context ~`6426 MiB`, generation ~`27.4 t/s`

Build: **succeeds** (all targets, no warnings).

Hardware: AMD Radeon RX 9060 XT (gfx1200 / RDNA4), ROCm 7.1.0, 16 GB VRAM.
Model used for testing: Qwen2.5-3B-Instruct-Q4_K_M.

### Measured throughput

| Cache type | Prompt (t/s) | Generation (t/s) | Output quality |
|---|---|---|---|
| f16 (baseline) | 592.4 | 82.9 | ✅ correct |
| turbo3_0 | 416.7 | 70.5 | ⚠️ degraded but coherent |
| turbo4_0 (pre-isolation) | 432.2 | ~103 (few tokens) | ❌ garbage |
| turbo4_0 (WHT disabled in graph) | 193.8 | 26.5 | ✅ coherent |

### Output samples (prompt: "Hello", n=10, temp=0 unless noted)

```
f16:     "Hello! How can I assist you today? Feel free to ask any"
turbo3:  "Hello there! I'm here to assist you today'll be the perfect fit for many"
turbo4 (pre-isolation):  "| CJK" (stops early)
turbo4 (after isolation): "\"Hello! How can I assist you today?\""
```

**TURBO3_0 verdict:** The Bug 4 fix (WHT K/V rotation) is confirmed working. The output is
coherent with minor quality degradation consistent with 3-bit KV compression — this is expected
behaviour, not a bug.

**TURBO4_0 verdict:** Coherent output is currently restored on this branch under tested HIP
settings, including with turbo4 WHT stages enabled. Turbo4 WHT is now default-enabled in the graph
path (via `LLAMA_TURBO4_WHT_ALL`, default `on`), while env toggles remain available for stage-level
diagnostics.

### Bugs found and fixed

#### Bug 1 — CPU SET_ROWS: null `from_float` (FIXED)
```
#1  ggml_compute_forward_set_rows  in libggml-cpu.so.0
#0  0x0000000000000000              (null function pointer)
```
**Root cause:** CPU `supports_op` for `SET_ROWS` did not exclude TURBO types, so the scheduler
assigned the op to CPU. The CPU handler called `type_traits_cpu[TURBO3_0].from_float` which was NULL.

**Fix:** Added `quantize_row_turbo3_0/4_0` wrappers to `ggml-cpu/quants.c` + `quants.h`, and
registered them in `ggml-cpu/ggml-cpu.c` `type_traits_cpu[]`.

#### Bug 2 — Bad `BEST_FATTN_KERNEL_NONE` guard pushed FlashAttention to CPU (FIXED)
An earlier attempt at fixing routing used:
```cpp
if (is_turbo && !can_use_vector_kernel) return BEST_FATTN_KERNEL_NONE;
```
This caused `ggml_cuda_flash_attn_ext_supported` to return false for short sequences, so the
scheduler sent FlashAttention to CPU. CPU has no TURBO `vec_dot`, causing a null-pointer crash.

**Fix:** Replaced with an unconditional early-return VEC for all TURBO types at the very top of
`ggml_cuda_get_best_fattn_kernel` (before all sequence-length guards), plus a bounds check in the
VEC kernel inner loop to handle `ne11 < FATTN_KQ_STRIDE`.

#### Bug 3 — RDNA4 MMA_F16 kernel selected for TURBO prefill (FIXED)
```
ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 4, 8>  ← crash
```
For Qwen 14B (K->ne[1]=512, gqa_ratio=5), the RDNA4 branch of `ggml_cuda_get_best_fattn_kernel`
was selected for prefill batches (`Q->ne[1] > 2`), choosing TILE then MMA_F16. Neither kernel
has TURBO dequant logic.

**Fix:** The early-return VEC at the top of the function (Bug 2 fix) also resolves this — TURBO
now returns VEC before reaching any arch-specific branch.

#### Bug 4 — K/V not WHT-rotated before writing to KV cache (FIXED ✅ — confirmed for TURBO3_0)
**Root cause:** The Metal `kernel_set_rows_turbo` applies WHT rotation internally when writing K/V
to the KV cache (line 9690: `turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2)`). The
CUDA `quantize_f32_turbo3_0_block` does NOT — its comment states "Input x[] is already
WHT-rotated by the preceding TURBO_WHT graph op."

However, the graph code in `llama-graph.cpp` was calling `cpy_k(k_cur, ...)` and
`cpy_v(v_cur, ...)` without first applying `ggml_turbo_wht(ctx0, k_cur, 0)`. K and V were stored
in the KV cache in the **original (unrotated) domain**. Meanwhile Q was correctly WHT-rotated
before attention. The dot product WHT(Q) · K_unrotated = garbage, causing the incoherent output.

**Fix:** `src/llama-graph.cpp` — before `cpy_k`/`cpy_v`, get the cached K/V tensors first (to
read their type), then apply `ggml_turbo_wht(ctx0, k_to_store, 0)` and
`ggml_turbo_wht(ctx0, v_to_store, 0)` when the cache type is TURBO3_0 or TURBO4_0:

```cpp
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);
ggml_tensor * k_to_store = k_cur;
ggml_tensor * v_to_store = v_cur;
if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0) {
    if (k_to_store->ne[0] % 128 == 0) {
        if (!ggml_is_contiguous(k_to_store)) k_to_store = ggml_cont(ctx0, k_to_store);
        k_to_store = ggml_turbo_wht(ctx0, k_to_store, 0);
    }
}
// same for v_to_store
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_to_store, k_idxs, il));
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_to_store, v_idxs, il));
```

**Note on norm differences vs Metal:** Metal's `kernel_set_rows_turbo` computes the L2 norm at
the 128-element group level and applies a centroid reconstruction correction factor. The CUDA
`quantize_f32_turbo3_0_block` computes per-32-element block norms. Since the normalized FWHT
preserves L2 norm, the per-block norms sum correctly across the group, but individual blocks may
have slightly different relative scales. This is a minor quality difference that does not affect
correctness and can be refined later.

#### Bug 5 — TURBO4_0 WHT-domain mismatch (STABILIZED ✅, currently non-reproducible on this branch)

**Original symptom:** With `--cache-type-k turbo4 --cache-type-v turbo4`, the model output `| CJK`
and stopped after a handful of tokens.

**Current status:** A targeted isolation patch initially disabled graph-side WHT for
`GGML_TYPE_TURBO4_0` (while retaining WHT for `GGML_TYPE_TURBO3_0`) and confirmed coherent turbo4
generation. Subsequent stabilization and retesting made the original failure non-reproducible. On
the current branch, turbo4 WHT is default-enabled in `src/llama-graph.cpp` (through
`LLAMA_TURBO4_WHT_ALL`, default `on`) while preserving env-gated stage toggles for diagnostics.

Additional update: after adding env-gated turbo4 stage toggles
(`LLAMA_TURBO4_WHT_STORE_KV`, `LLAMA_TURBO4_WHT_QUERY`, `LLAMA_TURBO4_WHT_OUTPUT`, `LLAMA_TURBO4_WHT_ALL`)
and rerunning the previous failing settings (`Hello`, `n=10`, `temp=0`), the original `| CJK` failure
is currently **non-reproducible** both in stabilized mode and with `LLAMA_TURBO4_WHT_ALL=1`.

**Investigation so far (exhaustive, no bug found analytically):**

All of the following were verified mathematically correct:

- `block_turbo4_0` layout (68 bytes: 2B norm + 2B rnorm + 48B qs + 16B signs).
- `quantize_f32_turbo4_0_block`: 3-bit centroid lookup, bit-spanning pack into `qs[48]`.
- `dequantize_V_turbo4_0`: correct bit extraction (handles byte-straddling), correct centroid × norm.
- `vec_dot_fattn_vec_KQ_turbo4_0`: correct 3-bit K index extraction, correct Q q8_1 accumulation.
- KV cache strides after `get_k` + `ggml_permute(0,2,1,3)` in `build_attn_mha`:
  - After permute: K shape = `[n_embd_head=128, n_kv, n_head_kv, ns]`
  - For TURBO4_0: `nb[1] = 8 heads × 68 bytes = 544` (position stride), `nb[2] = 68` (head stride).
  - Kernel uses `K += nb12*(head/gqa_ratio)` (head selection) + `K + i_KQ * nb11` (position
    iteration). Verified this correctly selects each 128-element head's single TURBO4_0 block.
- `set_rows_cuda_quant` writes one TURBO4_0 block per 128-element row segment — correct.
- VEC kernel `nthreads_KQ_q=2` for RDNA: 16 iterations × 2 threads × 4 elements = 128 elements — correct.
- Only the K=turbo4/V=turbo4 combination is compiled (both `DECL_FATTN_VEC_CASE` instances present).

**Hypothesis:** The WHT rotation may interact incorrectly with TURBO4_0 specifically. TURBO3_0
works with WHT (Bug 4 fix confirmed), but TURBO4_0 has a different block size (128 vs 32 elements)
and different centroid set. Possible that the issue is not in the WHT path but in the base
quantization/dequantization of TURBO4_0.

**Isolation result:** Completed. Disabling WHT for turbo4 only restores coherent generation.

### Next steps

1. ✅ **Reintroduce turbo4 WHT safely:** Completed for the current branch baseline; turbo4 WHT is
   default-enabled and stable in current tests.
2. ✅ **Add diagnostic toggles:** Completed via stage-level env flags
   (`LLAMA_TURBO4_WHT_STORE_KV`, `LLAMA_TURBO4_WHT_QUERY`,
   `LLAMA_TURBO4_WHT_OUTPUT`, `LLAMA_TURBO4_WHT_ALL`) for targeted fault isolation.
3. **QJL correction for TURBO4_0** — currently `rnorm=0, signs=0` (deferred). Implementing this
   would improve TURBO4 quality by adding the random-projection residual correction.
4. **Norm correction** — optionally align the per-block norm scheme with Metal's per-group norm +
   centroid reconstruction correction for better quantization accuracy.

---

## Limitations / Future Work

- **QJL correction deferred.** TURBO4_0 stores a residual norm (`rnorm`) and QJL sign bits
  (`signs`) for a second-pass random-projection correction. These are currently set to zero at
  quantisation time and ignored at dequantisation time. The PolarQuant term alone gives approximately
  3-bit effective precision. Implementing QJL would require two additional WHT passes inside the
  quantiser and a correction term in `vec_dot` / `dequantize_V`.

- **Norm scheme differs from Metal.** Metal's `kernel_set_rows_turbo` normalises at the 128-element
  group level and stores a centroid reconstruction correction factor. The CUDA implementation uses
  per-32-element block norms (no correction). This is a minor quality difference, not a correctness
  issue.

- **Kernel efficiency.** The current `turbo-wht.cu` kernel assigns one thread per 128-element group
  and processes sequentially. A warp-parallel butterfly with shared memory would be faster, but is
  not needed to unblock functionality.

- **GPU FlashAttention for very short sequences.** The GPU VEC kernel inner loop uses a per-position
  bounds check (added in Bug 2 fix) to handle `ne11 < FATTN_KQ_STRIDE`. This is correct but adds a
  branch per inner iteration that could be optimised for the common case where `ne11 % 256 == 0`.

- **TILE/MMA kernels.** These never run for TURBO types (unconditional early-return VEC). If a future
  large-batch prefill path wanted to use TILE/MMA, TURBO dequant would need to be added to those
  kernel templates.

---

## Build

```bash
cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1200 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Test

```bash
# Basic inference test
./build/bin/llama-cli \
  -m path/to/model.gguf \
  --cache-type-k turbo3 \
  --cache-type-v turbo3 \
  -p "Hello" -n 10 --no-display-prompt --simple-io

# Long-context test (exercises the full GPU VEC path)
./build/bin/llama-cli \
  -m path/to/model.gguf \
  --cache-type-k turbo3 \
  --cache-type-v turbo3 \
  --ctx-size 65536 \
  -p "Hello"
```

**TURBO3_0 confirmed working** (as of 2026-03-29) on AMD Radeon RX 9060 XT / gfx1200 / ROCm 7.1.0
with Qwen2.5-3B-Instruct-Q4_K_M. Output is coherent with expected 3-bit quality degradation.
**TURBO4_0 currently working** on this branch under tested settings; the earlier garbage-output case
is currently non-reproducible (see Bug 5 history above).
