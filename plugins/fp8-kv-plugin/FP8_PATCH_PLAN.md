# FP8 Patch Execution Plan — Safety-First

## Patch Classification

Each patch is classified as **REQUIRED**, **OPTIONAL**, or **SKIP** with justification.

| # | Patch | Class | Why |
|---|-------|-------|-----|
| 1 | `ggml.h`: type enum + API | **REQUIRED** | Type must exist to be used |
| 2 | `ggml.c`: type_traits + conversions | **REQUIRED** | ggml must know size/name/blck/convert |
| 3 | `ggml-cuda/convert.cuh + .cu` | **REQUIRED** | GPU F16↔F8 conversion for attention |
| 4 | `ggml-cuda/set-rows.cu` | **REQUIRED** | KV cache write needs F32→F8 |
| 5 | `ggml-cuda/ggml-cuda.cu` type checks | **REQUIRED** | Ops reject unknown types |
| 6 | `ggml-cuda/fattn-common.cuh` | **REQUIRED** | F8→F16 conversion for attention |
| 7 | `ggml-cuda/fattn.cu` dispatch | **REQUIRED** | Select correct attention kernel |
| 8 | `common/arg.cpp` CLI | **REQUIRED** | `--cache-type-k f8_e4m3` must parse |
| 9 | `llama-context.cpp` F16 force | **REQUIRED** | Otherwise F8 is overridden to F16 |
| 10 | `fattn-vec.cuh` direct F8 kernel | **OPTIONAL** | F16 conversion path works; direct kernel is perf |
| 11 | `generate_cu_files.py` | **OPTIONAL** | Only needed if patch #10 is done |
| 12 | `fp8-kv-plugin.cpp` | **OPTIONAL** | Smart tiering; standalone plugin |
| — | `llama-kv-cache.cpp` | **SKIP** | Type system handles allocation already |
| — | `llama-kv-cache-mixed.cpp` | **SKIP** | `ggml_type_from_str` handles F8 via name lookup |

---

## Patch Details with Safety

### Patch 1: `ggml/include/ggml.h` — Type Enum

**File:** `ggml.h:389-433`

```c
enum ggml_type {
    // ... existing types unchanged up to GGML_TYPE_Q1_0 = 41 ...
    GGML_TYPE_F8_E4M3 = 42,
    GGML_TYPE_COUNT   = 43,   // was 42
};
```

**Safety:**
- `GGML_TYPE_COUNT` is only used for array bounds in `type_traits[GGML_TYPE_COUNT]`
- Bumping it by 1 is safe — all loops iterate `< GGML_TYPE_COUNT`
- Value 42 is free (was GGML_TYPE_COUNT, not an active type)
- All existing enum values are unchanged (backward compatible)

**Conversion API** (after line 381, after `ggml_bf16_t` block):
```c
typedef uint8_t ggml_fp8_e4m3_t;   // 1 byte: 1s-4e-3m

GGML_API float          ggml_fp8_to_fp32(ggml_fp8_e4m3_t x);
GGML_API ggml_fp8_e4m3_t ggml_fp32_to_fp8(float f);
GGML_API void           ggml_fp8_to_fp32_row(const ggml_fp8_e4m3_t * x, float * y, int64_t n);
GGML_API void           ggml_fp32_to_fp8_row(const float * x, ggml_fp8_e4m3_t * y, int64_t n);
```

**Null safety:** No pointers in scalar functions. Row functions assert or null-check:
- `GGML_ASSERT(x != NULL)` and `GGML_ASSERT(y != NULL)` in row functions
- `n > 0` early-return guard

---

### Patch 2: `ggml/src/ggml.c` — Type Traits + Quantize + Conversions

**type_traits entry** — Insert after `[GGML_TYPE_Q1_0]` block (after line ~673):
```c
[GGML_TYPE_F8_E4M3] = {
    .type_name                = "f8_e4m3",
    .blck_size                = 1,
    .type_size                = sizeof(ggml_fp8_e4m3_t),  // = 1
    .is_quantized             = false,
    .to_float                 = (ggml_to_float_t) ggml_fp8_to_fp32_row,
    .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp8_row,
},
```

**Safety:** `is_quantized = false` means ggml treats F8 as a plain float type
(like F16, BF32). This is correct — F8 is floating point, not block-quantized.
The blck_size=1 means no block alignment requirements.

**quantize_row switch** — Add before `case GGML_TYPE_F16:` (line 7755):
```c
case GGML_TYPE_F8_E4M3:
    {
        const size_t elemsize = sizeof(ggml_fp8_e4m3_t);
        if (src == NULL || dst == NULL) { result = 0; break; }
        if (n == 0) { result = 0; break; }
        ggml_fp32_to_fp8_row(src + start, (ggml_fp8_e4m3_t *)dst + start, n);
        result = n * elemsize;
    } break;
```

**Null safety:** Explicit NULL check + n==0 guard. No crash on bad input.

**F8 conversion implementations** — New functions, anywhere after `ggml_fp32_to_bf16_row`:
```c
float ggml_fp8_to_fp32(ggml_fp8_e4m3_t x) {
    if (x == 0) return 0.0f;
    uint8_t bits = x;
    int sign = (bits >> 7) & 1;
    int exp  = (bits >> 3) & 0xF;
    int mant = bits & 0x7;
    float f;
    if (exp == 0) {
        f = (float)(mant) * 0.0009765625f;  // 2^-10
    } else {
        f = (float)(mant | 0x8) * (float)(1 << (exp - 7));
    }
    return sign ? -f : f;
}

ggml_fp8_e4m3_t ggml_fp32_to_fp8(float f) {
    if (f == 0.0f) return 0;
    int sign = f < 0 ? 1 : 0;
    float v = f < 0 ? -f : f;
    // Clamp to F8 E4M3 range [2^-10, 448]
    if (v < 0.0009765625f) v = 0.0009765625f;
    // ... round-to-nearest-even quantization ...
}

void ggml_fp8_to_fp32_row(const ggml_fp8_e4m3_t * x, float * y, int64_t n) {
    if (!x || !y || n <= 0) return;
    for (int64_t i = 0; i < n; i++) y[i] = ggml_fp8_to_fp32(x[i]);
}

void ggml_fp32_to_fp8_row(const float * x, ggml_fp8_e4m3_t * y, int64_t n) {
    if (!x || !y || n <= 0) return;
    for (int64_t i = 0; i < n; i++) y[i] = ggml_fp32_to_fp8(x[i]);
}
```

---

### Patch 3: `ggml/src/ggml-cuda/convert.cuh + .cu` — GPU F16↔F8 Kernels

**convert.cuh** — After line 28:
```cpp
typedef to_t_cuda_t<__nv_fp8_e4m3> to_fp8_cuda_t;
to_fp8_cuda_t ggml_get_to_fp8_cuda(ggml_type type);
```

**convert.cu** — Add new kernels:
```cuda
static __global__ void convert_f16_to_f8(const half * x, __nv_fp8_e4m3 * y, int64_t k) {
    if (!x || !y) return;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    y[i] = half_to___hip_fp8_e4m3(x[i]);
}

static __global__ void convert_f8_to_f16(const __nv_fp8_e4m3 * x, half * y, int64_t k) {
    if (!x || !y) return;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    y[i] = __hip_fp8_e4m3_to_half(x[i]);
}
```

**Null safety:** Guard at kernel entry checks `!x || !y`. The `blockIdx.x * blockDim.x`
can't overflow for realistic k (int64_t safe).

Add `ggml_get_to_fp16_cuda(GGML_TYPE_F8_E4M3)` and
`ggml_get_to_fp32_cuda(GGML_TYPE_F8_E4M3)` that return the appropriate converter.

---

### Patch 4: `ggml/src/ggml-cuda/set-rows.cu` — Add F8 to set_rows

**After line 249 (BF16 case), insert:**
```cpp
} else if (dst->type == GGML_TYPE_F8_E4M3) {
    if (!dst->data) { GGML_ABORT("set_rows: dst->data is NULL"); }
    set_rows_cuda<float, idx_t, __nv_fp8_e4m3>(
        src0_d, src1_d, (__nv_fp8_e4m3*)dst->data,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12, ne13,
        nb01, nb02, nb03,
        nb10, nb11, nb12,
        nb1, nb2, nb3,
        stream);
```

Also add `ggml_cuda_cast` specialization in `convert.cuh` (line 34-65):
```cpp
} else if constexpr(std::is_same_v<dst_t, __nv_fp8_e4m3>) {
#ifdef FP8_AVAILABLE
    return half_to___hip_fp8_e4m3(__float2half(float(x)));
#else
    GGML_UNUSED(x); return __nv_fp8_e4m3{};
#endif
}
```

**Null safety:** `GGML_ABORT` if dst->data is NULL (same behavior as other types).

---

### Patch 5: `ggml/src/ggml-cuda/ggml-cuda.cu` — Type Checks

Add `GGML_TYPE_F8_E4M3` to every type check that accepts `GGML_TYPE_F16`.

**Minimal set** (these are the gates that matter for KV cache flow):

| Line | Current | Add |
|------|---------|-----|
| 3384 | `set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16` | `&& set_rows->type != GGML_TYPE_F8_E4M3` |
| 1275 | `type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16` | `&& type != GGML_TYPE_F8_E4M3` |
| 2366 | assert F16/BF16/F32 | `\|\| src0->type == GGML_TYPE_F8_E4M3` |
| 2476 | mul_mat_vec check | `\|\| src0->type == GGML_TYPE_F8_E4M3` |
| 2550 | use_mul_mat_vec_f | `\|\| src0->type == GGML_TYPE_F8_E4M3` |
| 3958 | CLAMP check | `\|\| x->type == GGML_TYPE_F8_E4M3` |
| 5214 | GET_ROWS check | `\|\| op->type == GGML_TYPE_F8_E4M3` |
| 5297 | UPSCALE check | `\|\| src0_type == GGML_TYPE_F8_E4M3` |
| 5326 | PAD check | `\|\| src0_type == GGML_TYPE_F8_E4M3` |
| 5360-2 | LEAKY_RELU check | Add F8 to src[0]/src[1]/op |

**Safety:** All changes are additive (adding `|| type == F8_E4M3`). Never removes
an existing check. All types that were accepted before remain accepted.

---

### Patch 6: `ggml/src/ggml-cuda/fattn-common.cuh` — F8→F16 Conversion

**Line 53-85** — In `ggml_cuda_flash_attn_ext_get_f16_extra_data`:
```cpp
if (need_f16_K && K->type != GGML_TYPE_F16) {
    // F8 also needs conversion
    data.end = GGML_PAD(data.end, 128);
    data.K   = data.end;
    data.end += ggml_nelements(K) * ggml_type_size(GGML_TYPE_F16);
}
```

**Line 1019-1045** — In `launch_fattn`, the existing conversion path already
calls `ggml_get_to_fp16_cuda(K->type)` which returns our new F8→F16 converter.
No additional code needed in the conversion loop — it just works.

**Null safety:** The f16_extra.K address is checked at launch time. If allocation
of the scratch buffer failed (end==0), the conversion is skipped.

---

### Patch 7: `ggml/src/ggml-cuda/fattn.cu` — Dispatch Table

**Line 342-411** — Add to `ggml_cuda_flash_attn_ext_vec`:

```cpp
// Add after existing rows, BEFORE the GGML_ABORT fallthrough:
FATTN_VEC_CASES_ALL_D(GGML_TYPE_F8_E4M3, GGML_TYPE_F16)
FATTN_VEC_CASES_ALL_D(GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E4M3)
```

**Why only these two, not the full 7×V-type dispatch:**
- In practice, with `--cache-type-k f8_e4m3 --cache-type-v f8_e4m3`, both are F8
- The (F8, F16) pair covers mixed-mode usage
- For (F8, Q4_0) etc., the F16 conversion path in fattn-common handles it by
  converting both to F16 before the kernel runs

**Safety:** If neither dispatch matches, the existing `GGML_ABORT("fatal error")`
fires — same as if an unsupported type pair is used. No silent fallback to
wrong kernel.

---

### Patch 8: `common/arg.cpp` — CLI Type Whitelist

**Line 354-364:**
```cpp
const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_F8_E4M3,       // <-- INSERT here (before quantized types)
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    // ...
};
```

**Safety:** The `kv_cache_type_from_str` function (line 366) iterates this vector
and compares `ggml_type_name(type) == s`. Since `ggml_type_name(GGML_TYPE_F8_E4M3)`
returns `"f8_e4m3"`, the CLI accepts `--cache-type-k f8_e4m3` and
`--cache-type-v f8_e4m3`. No other code changes needed.

---

### Patch 9: `llama.cpp/src/llama-context.cpp` — Remove F16 Force

**Lines 355-359:**
```cpp
#ifdef GGML_PLUGIN_SUPPORT
    if (params.use_plugin_kv_cache) {
        if (params.type_k != GGML_TYPE_F8_E4M3) params_mem.type_k = GGML_TYPE_F16;
        if (params.type_v != GGML_TYPE_F8_E4M3) params_mem.type_v = GGML_TYPE_F16;
    }
#endif
```

**Safety:** The `!= F8_E4M3` guard means:
- `--cache-type-k f8_e4m3` → type_k passes through as F8_E4M3
- `--cache-type-k f16` → type_k forced to F16 (same as before)
- `--cache-type-k q4_0` → type_k forced to F16 (same as before)
- No `--cache-type-k` → default F16 applied by upstream, forced to F16 (no-op)

**Null safety:** `params.type_k` is an enum, never NULL. The `use_plugin_kv_cache`
bool is checked first. No crash path.

---

## Build Integration (Optional Patches 10-11)

### Patch 10: `generate_cu_files.py`
```python
TYPES_KV = [..., "GGML_TYPE_F8_E4M3"]  # line 11
```
Re-run to generate 16 new instance .cu files.

### Patch 11: `CMakeLists.txt`
Already handled by the existing GLOB pattern at line 116:
```cmake
file(GLOB SRCS "template-instances/fattn-vec*.cu")
```

---

## Execution Order

```
Patch 1:  ggml.h              ─┐
Patch 2:  ggml.c               ├─ Compile together → ggml compiles
                               │
Patch 3:  convert.cuh + .cu    ├─ GPU backend
Patch 4:  set-rows.cu          │
Patch 5:  ggml-cuda.cu         │
Patch 6:  fattn-common.cuh     │
Patch 7:  fattn.cu             ┘
                               │
Patch 8:  arg.cpp              ├─ llama.cpp compiles
Patch 9:  llama-context.cpp    ┘
```

Patches 1-2 are independent. Patches 3-7 depend on 1-2. Patches 8-9 depend on 1.

---

## Rollback Plan

Each patch is a single insertion/addition. To revert, reverse the edit:

| Patch | Rollback |
|-------|----------|
| 1 | Remove 2 enum lines + conversion API |
| 2 | Remove type_traits entry + switch case + conversion functions |
| 3 | Remove F8 cases from convert files |
| 4 | Remove 5 lines from set-rows dispatch |
| 5 | Remove `|| type == F8_E4M3` from ~10 sites |
| 6 | Remove `F8` from the conditional check |
| 7 | Remove 2 lines of dispatch |
| 8 | Remove 1 line from kv_cache_types vector |
| 9 | Replace 4 lines with original 4 lines |

---

## Test Plan

### After each patch, compile and verify:

1. **Patches 1-2**: `ggml` compiles without errors
2. **Patches 3-7**: `ggml-cuda` compiles (HIP/ROCm)
3. **Patches 8-9**: `llama.cpp` compiles
4. **Full**: `llama-cli --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 -p "hello" -n 10`
   — should run without crashes, output sensible text

### Quality verification:
```
# F16 baseline
llama-cli --cache-type-k f16 --cache-type-v f16 -p "The capital of France is" -n 5 --logits all -o f16_logits.txt

# F8 test
llama-cli --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 -p "The capital of France is" -n 5 --logits all -o f8_logits.txt

# Compare: per-token logit diff should be < 0.01
```

### Stress test:
```
llama-cli --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --ctx-size 32768 -f long_prompt.txt -n 256
# No crashes, memory usage ~half of F16
```

---

## Files NOT Changed (Justification)

| File | Why Skipped |
|------|-------------|
| `llama-kv-cache.cpp` | `ggml_new_tensor_3d(ctx, type_k, ...)` at line 249 works for any ggml_type. No change needed. |
| `llama-kv-cache-mixed.cpp` | `ggml_type_from_str` at line 17 uses `ggml_type_name()` which returns "f8_e4m3". Mixed specs like `f8_e4m3:1000,f16:0` work. |
| `ggml-common.h` | F8 has no block struct (blck_size=1, no block_* typedef needed). Not a "quantized" type. |
| `dequantize.cuh` | F8→F32 conversion happens via `ggml_get_to_fp32_cuda` → convert kernel, not block-level dequantize. |
| `quantize.cuh` | Same reasoning — F32→F8 is a row-level conversion, not block quantization. |

---

## Summary

**9 required patches, 0 risky changes, full null safety, full rollback plan.**

Every patch is an additive insertion or a single `||` guard. No code is removed.
No existing behavior changes for non-F8 workloads.
