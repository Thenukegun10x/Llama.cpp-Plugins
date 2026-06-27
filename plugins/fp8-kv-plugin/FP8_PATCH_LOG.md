# FP8 Patch Log — Final State

## Build Status

| Component | Status |
|-----------|--------|
| `llama.cpp` (HIP + ggml) | ✅ Compiles and links |
| `smart-tq-plugin.dll` | ✅ Compiles and links |
| F8 + CLI `--no-warmup` | ✅ Works |
| F8 + server `--no-warmup` | ⚠️ Hangs (plugin cache-clear loop) |
| F8 without `--no-warmup` | ❌ Crashes (warmup context buffer assertion) |

**Bottom line**: F8 type system and GPU kernels are correct. F8 inference works on the main context. The warmup/second-context buffer allocation is a known issue — the `hparams.no_alloc` flag + dummy buffer assignment path doesn't handle F8 tensor sharing correctly. Use `--no-warmup` as a workaround.

---

## All Patches Applied

### 1. Type System

| File | Change |
|------|--------|
| `ggml/include/ggml.h:431-433` | Add `GGML_TYPE_F8_E4M3 = 42`, bump `GGML_TYPE_COUNT` to 43 |
| `ggml/include/ggml.h:382+` | Add `ggml_fp8_e4m3_t` typedef + conversion API |
| `ggml/src/ggml.c:666-680` | Add `type_traits[F8_E4M3]` with blck=1, size=1, is_quantized=false |
| `ggml/src/ggml.c:7755+` | Add F8 case to `quantize_row` |
| `ggml/src/ggml.c:508+` | Implement `ggml_fp8_to_fp32` / `ggml_fp32_to_fp8` / row functions |
| `ggml/src/ggml-cpu/ggml-cpu.c:406` | Add `type_traits_cpu[F8_E4M3]` with `from_float = ggml_fp32_to_fp8_row` |

### 2. GPU Conversion Kernels

| File | Change |
|------|--------|
| `ggml-cuda/convert.cuh` | Add `ggml_cuda_cast<uint8_t>` (F32→F8 encode) and `uint8_t→half/float` (F8→F16/F32 decode) specializations — pure arithmetic, no HIP intrinsics |
| `ggml-cuda/convert.cu` | Add `dequantize_f8_to_f16_cuda` and `dequantize_f8_to_f32_cuda` kernels (proper `half*`/`float*` output stride) |
| `ggml-cuda/convert.cu` | Wire F8→F16/F32 converters into `ggml_get_to_fp16_cuda` / `ggml_get_to_fp32_cuda` |

### 3. GPU set_rows (KV cache write)

| File | Change |
|------|--------|
| `ggml-cuda/set-rows.cu:250-259` | Add F8_E4M3 dispatch using `set_rows_cuda<float, idx_t, uint8_t>` |

### 4. GPU Flash Attention (KV cache read)

| File | Change |
|------|--------|
| `ggml-cuda/fattn-common.cuh:68-91` | Alloc F8→F16 scratch buffer in flash attention |
| `ggml-cuda/fattn-common.cuh:651,674` | Add F8 case to `get_dequantize_K/V` (returns nullptr — unreached, F8→F16 done upstream) |
| `ggml-cuda/fattn-vec.cuh:540-541` | Set `need_f16_K/V = true` for F8 |
| `ggml-cuda/fattn-vec.cuh:595-610` | Add extern decls for F8_E4M3 (64, 128, 256 head dims) |
| `ggml-cuda/fattn.cu:350-418` | Add F8 dispatch rows (F8×F16, F8×BF16, all-V×F8, F8×F8) |
| `ggml-cuda/fattn.cu:655-656` | Set `need_f16_K/V = true` for F8 in VEC allocation |
| `template-instances/generate_cu_files.py:11` | Add `GGML_TYPE_F8_E4M3` to `TYPES_KV` |
| 15 generated `.cu` files | `fattn-vec-instance-*-f8_e4m3.cu` etc. |

### 5. GPU MulMat (non-flash attention fallback)

| File | Change |
|------|--------|
| `ggml-cuda/ggml-cuda.cu:2379` | F8→F16 scratch before hipBLAS via `ggml_get_to_fp16_cuda(F8)` |

### 6. GPU Type Checks

| File | Change |
|------|--------|
| `ggml-cuda/ggml-cuda.cu:1275` | graph capture: accept F8_E4M3 |
| `ggml-cuda/ggml-cuda.cu:2366` | mul_mat assert: accept F8_E4M3 |
| `ggml-cuda/ggml-cuda.cu:2479,2553` | mul_mat_vec support: accept F8_E4M3 |
| `ggml-cuda/ggml-cuda.cu:3387` | rope-set_rows fusion: **exclude** F8 (ROPE can't fuse with F8) |
| `ggml-cuda/ggml-cuda.cu:5218` | SET_ROWS op support: accept F8_E4M3 |

### 7. CLI + Context

| File | Change |
|------|--------|
| `common/arg.cpp:357` | Add `GGML_TYPE_F8_E4M3` to `kv_cache_types` |
| `llama-context.cpp:355-359` | Don't force F16 when plugin active AND type is F8_E4M3 |
| `llama-context.cpp:361-364` | Add diagnostic log for KV cache type |

### 8. Plugin (smart-tq)

| File | Change |
|------|--------|
| `smart-tq-plugin.cpp` | Add `g_fp8_mode` env var (`SMART_KV_FP8=1`), `fp8_active` flag, VRAM budget adjusted (1 B/el), memory_capacity doubled for F8, stats report `fp8=1` |

### 9. Build + Usage

| File | Change |
|------|--------|
| `start.bat` | Add `fp8` / `f8` / `smart-fp8` profiles with `--cache-type-k f8_e4m3 --cache-type-v f8_e4m3`, `SMART_KV_FP8=1`, `--no-warmup` |

---

## Bugs Found and Fixed

### Bug 1: F8 decode 8x magnitude error (CRITICAL)
**Files**: `convert.cuh:112-113,126-127` and `ggml.c:527`
**Root cause**: `float(mant | 0x8) * exp2f(exp - 7)` should be `(1.0f + mant/8.0f) * exp2f(exp - 7)`. The significand was being treated as raw integer (8-15) instead of normalized (1.0-1.875). All decoded F8 values were 8x too large.
**Fix**: Changed to `(1.0f + float(mant) * 0.125f) * exp2f(float(f8_exp - 7))`.

### Bug 2: CPU subnormal decode 2x too small (MEDIUM)
**File**: `ggml.c:524`
**Root cause**: Used `mant * 0.0009765625f` (mant/1024) but encoder uses `mant * 512.0f`. Should be `mant * 0.001953125f` (mant/512) to round-trip correctly.
**Fix**: Changed constant.

### Bug 3: `frexpf` host-only from GPU device code (CRITICAL)
**File**: `convert.cuh:78`
**Root cause**: `frexpf` is not available in CUDA/HIP device math libraries.
**Fix**: Replaced with F32 bit manipulation via union `{ float f; uint32_t u; }`.

### Bug 4: `memcpy` host-only from GPU device code (CRITICAL)
**File**: `convert.cuh:73` (later removed)
**Root cause**: `memcpy` is a host-only CRT function.
**Fix**: Replaced with union type-punning.

### Bug 5: `1 << (exp - 7)` UB when exp < 7 (CRITICAL)
**Files**: `convert.cuh:101,115` (old code, now replaced)
**Root cause**: Left-shift by negative value is UB in C. On AMD GPU, shifts wrap around.
**Fix**: Replaced with `exp2f(float(exp - 7))`.

### Bug 6: `__nv_fp8_e4m3` HIP type uses CDNA-specific intrinsics on gfx1201 (CRITICAL)
**File**: `convert.cu:839` (removed)
**Root cause**: `__hip_fp8_e4m3::operator __half()` uses `__builtin_amdgcn_cvt_pk_fp8_f32` which is CDNA-specific, not available on gfx1201 (RDNA4). Produces "device kernel image is invalid".
**Fix**: Replaced ALL `__nv_fp8_e4m3` usage with `uint8_t` + pure arithmetic in all GPU kernels.

### Bug 7: `convert_unary_cont_cuda<uint8_t>` writes `half` at `uint8_t` strides (CRITICAL)
**File**: `convert.cu:765`
**Root cause**: When converting F8→F16 for flash attention, the converter was writing `half` values (2 bytes) at `uint8_t` strides (1 byte). Adjacent `half` values overlapped, the last write overflowed the buffer → access violation.
**Fix**: Created dedicated `dequantize_f8_to_f16_cuda` kernel that reads `uint8_t*` and writes `half*` with proper 2-byte strides.

### Bug 8: CPU `type_traits_cpu[F8_E4M3]` missing → `from_float = NULL` (CRITICAL)
**File**: `ggml-cpu/ggml-cpu.c:406`
**Root cause**: The CPU backend has a separate `type_traits_cpu` array. Missing entry → `from_float = NULL` → `ggml_compute_forward_set_rows_f32` calls NULL function pointer → access violation.
**Fix**: Added entry with `from_float = ggml_fp32_to_fp8_row`.

### Bug 9: iGPU can't load gfx1201-compiled kernels
**Root cause**: The iGPU (AMD Radeon(TM) Graphics) doesn't support RDNA4 instruction set. When `HIP_VISIBLE_DEVICES` is not set, the HIP runtime tries to load kernels on all devices including the iGPU.
**Fix**: `start.bat` already has `set HIP_VISIBLE_DEVICES=1`. When running manually, set it explicitly.

### Bug 10: `ggml_backend_buft_alloc_buffer` for ROCm0 line not guarded
**File**: `llama-context.cpp:355-359`
**Root cause**: When `use_plugin_kv_cache` is true, the code forces `type_k = GGML_TYPE_F16` unconditionally.
**Fix**: Guard so F8_E4M3 passes through: `if (params.type_k != GGML_TYPE_F8_E4M3) params_mem.type_k = GGML_TYPE_F16;`

---

## Known Issues (NOT Fixed)

### Issue A: Warmup/second-context crash — `GGML_ASSERT(tensor->buffer == NULL)`
**Symptom**: `llama.cpp/ggml/src/ggml-backend.cpp:1994: GGML_ASSERT(tensor->buffer == NULL) failed`
**When**: Second `llama_kv_cache::init()` call (warmup context or slot context) tries to allocate F8_E4M3 tensors that already have buffers from the first context.
**Root cause analysis**: The `hparams.no_alloc` flag for the Qwen3.5 hybrid model is `true` for the attention cache, causing the init code at line 296-299 to assign a dummy buffer to all tensors. When the second context's init runs (sharing layers with the first), `ggml_tallocr_alloc` finds `tensor->buffer != NULL` → assertion fails.
**Workaround**: `--no-warmup` (included in `start.bat fp8`).
**Needed fix**: Either:
  - Skip already-buffered tensors in `ggml_tallocr_alloc` (tried — causes allocation to return NULL → downstream crash)
  - Or not assign dummy buffers when the context uses real GPU allocation
  - Or make the warmup context not create independent F8 tensors

### Issue B: Warmup/second-context crash — `GGML_ASSERT(base == nullptr)` in memory_breakdown
**Symptom**: `llama-kv-cache.cpp:779`
**When**: After the buffer allocation, `memory_breakdown()` checks `hparams.no_alloc && base == nullptr`. When a real buffer exists with `no_alloc=true`, this fails.
**Workaround**: None needed if Issue A is fixed/worked-around.

### Issue C: Smart-tq plugin cache-clear loop with FP8
**Symptom**: Server loads but plugin's `kv_cache_clear` is called repeatedly (16+ times) → eventual crash.
**When**: Only with smart-tq plugin + F8 cache type + server (not CLI).
**Root cause**: Unknown. Possibly related to the 0-size placeholder buffer or the FP8 tensor type confusing the plugin's layer management.
**Workaround**: Use `--no-warmup --parallel 1` with the plugin.

---

## Working Commands

### CLI test (works with `--no-warmup`)
```bash
set HIP_VISIBLE_DEVICES=1
llama-cli -m model.gguf -ngl 99 -fa on --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup -c 256 -n 20 -p "Hello"
```

### Server test (needs `--no-warmup --parallel 1` + plugin)
```bash
set HIP_VISIBLE_DEVICES=1
set SMART_KV_FP8=1
set SMART_KV_PROFILE=ultra
llama-server -m model.gguf -ngl 99 -fa on --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --plugin-kv-cache on --no-warmup -c 512 -b 256 -ub 128 --parallel 1 --host 0.0.0.0 --port 12345
```

### Using start.bat
```
start.bat fp8
```

---

## Files Modified (Complete List)

| File | Changes |
|------|---------|
| `ggml/include/ggml.h` | Type enum + conversion API |
| `ggml/src/ggml.c` | type_traits, quantize_row, F8↔F32 conversion functions |
| `ggml/src/ggml-cpu/ggml-cpu.c` | type_traits_cpu entry for F8_E4M3 |
| `ggml/src/ggml-cuda/convert.cuh` | F8 cast specializations (encode + 2× decode paths) |
| `ggml/src/ggml-cuda/convert.cu` | F8→F16/F32 dedicated kernel + converter registration |
| `ggml/src/ggml-cuda/set-rows.cu` | F8 set_rows dispatch |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 7 type protection + mul_mat F8→F16 conversion |
| `ggml/src/ggml-cuda/fattn-common.cuh` | F8→F16 scratch allocation + dequantize stubs |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | need_f16_K/V for F8 + extern decls |
| `ggml/src/ggml-cuda/fattn.cu` | F8 dispatch rows + VEC allocation |
| `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` | TYPES_KV |
| 15 generated `.cu` files | fattn-vec-instance-f8_e4m3-*.cu |
| `common/arg.cpp` | kv_cache_types |
| `llama-context.cpp` | F16 force guard + diagnostic log |
| `smart-tq-plugin.cpp` | FP8 mode toggle + memory_capacity doubling |
| `fp8-kv-plugin/fp8-kv-plugin.cpp` | Standalone FP8 plugin (reference/scaffold) |
| `start.bat` | fp8/f8/smart-fp8 profiles |

---

## Unchanged Files (No Patches Needed)

| File | Reason |
|------|--------|
| `llama-kv-cache.cpp` | Allocation works via ggml type system; no F8-specific changes needed |
| `llama-kv-cache-mixed.cpp` | `ggml_type_from_str` uses `ggml_type_name()` which returns "f8_e4m3" |
| `ggml-common.h` | F8 has no block struct (blck_size=1) |
| `dequantize.cuh` | F8→F32 is row-level, not block dequantize |
| `quantize.cuh` | Same |
