# Mixed KV Cache — Implementation Plan

## Overview

Replace the single K/V tensor with **3 precision tiers** selected by token age:

```
position age:   newest ... 1K ... 6K ... oldest
                ┌──────────┬──────────┬──────────┐
                │  Q8_0    │  Q4_0    │  Q2_K    │
                │ bucket 0 │ bucket 1 │ bucket 2 │
                └──────────┴──────────┴──────────┘
```

All tokens start in bucket 0. After 1000 new tokens, a token re-quantizes to bucket 1. After 6000 total, to bucket 2.

---

## Files to create

### `src/llama-kv-cache-mixed.h` (~80 lines)

New class `llama_kv_cache_mixed : public llama_memory_i`.

Stores N `llama_kv_cache` sub-caches (one per bucket) plus a `bucket_config` array:

```cpp
struct bucket {
    ggml_type type;
    uint32_t  capacity;  // 0 = "the rest"
};

struct kv_cache_mixed_buckets {
    int        n_buckets;
    bucket     buckets[4];
};
```

```cpp
class llama_kv_cache_mixed : public llama_memory_i {
    llama_kv_cache_mixed(
        const llama_model & model,
        const llama_hparams & hparams,
        const kv_cache_mixed_buckets & bucket_cfg,
        ggml_type type_default,    // fallback if buckets disabled
        bool v_trans, bool offload, bool unified,
        uint32_t kv_size, uint32_t n_seq_max,
        uint32_t n_pad, uint32_t n_swa,
        llama_swa_type swa_type,
        llama_memory_t mem_other,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse,
        const layer_share_cb & share);
};
```

Sub-caches stored as `std::vector<std::unique_ptr<llama_kv_cache>> sub_caches`.

### `src/llama-kv-cache-mixed.cpp` (~300 lines)

**Constructor:**
1. For each bucket, create a chain filter
2. Create a `llama_kv_cache` sub-cache for each bucket with its own `type_k`/`type_v`
3. Each sub-cache gets `bucket_cfg.capacity` slots (or remaining for last bucket)

**`get_bucket_for_pos(pos, head_pos)`** — static helper:
```
age = (head_pos - pos + kv_size) % kv_size
cumulative = 0
for each bucket:
    if age < cumulative + bucket.capacity: return bucket_index
    cumulative += bucket.capacity
```

**`get_k(il, n_kv, sinfo)`** — returns the K tensor view:
```
// Build a merged view: copy from sub-caches into a temporary F16 tensor
// Handles the case where the attention range spans multiple buckets
```

**`get_v(il, n_kv, sinfo)`** — same pattern.

**`cpy_k(ctx, k_cur, k_idxs, il, sinfo)`**:
1. Write `k_cur` to sub_cache[0] (bucket 0 = highest precision)
2. For each evicted position (those aging out of the previous bucket):
   - `ggml_cpy(ctx, src_tensor, dst_tensor)` to re-quantize

**`cpy_v`** — same pattern.

**Migration logic** in `set_input_k_idxs` / `apply()`:
- Track which positions cross a bucket boundary per write
- Add re-quant `ggml_cpy` nodes to the graph for those positions

---

## Files to patch

### 1. `include/llama.h` — Add params

After `type_k` / `type_v` (line 368):

```c
// mixed KV cache: "--cache-type-k-mixed q8_0:1000,q4_0:5000,q2_k:0"
struct llama_cache_type_mixed {
    const char * types;   // colon-separated spec string
};
```

Add default in `llama_context_default_params`:
```c
/*.cache_type_k_mixed       =*/ {nullptr},
/*.cache_type_v_mixed       =*/ {nullptr},
```

### 2. `common/common.h` — Add field

After `cache_type_k` / `cache_type_v`:
```cpp
std::string cache_type_k_mixed;  // "q8_0:1000,q4_0:5000,q2_k:0"
std::string cache_type_v_mixed;
```

### 3. `common/common.cpp` — Pass through

After `cparams.type_k = params.cache_type_k`:
```cpp
cparams.cache_type_k_mixed = params.cache_type_k_mixed;
cparams.cache_type_v_mixed = params.cache_type_v_mixed;
```

### 4. `common/arg.cpp` — CLI arg

Add after `--cache-type-k` (line 2088):
```cpp
{"-ctkm", "--cache-type-k-mixed"}, "SPEC",
"mixed KV cache for K: e.g. q8_0:1000,q4_0:5000,q2_k:0",
[](common_params & params, const std::string & value) {
    params.cache_type_k_mixed = value;
}
```

Same for `--cache-type-v-mixed`.

### 5. `src/llama-context.cpp` — Memory params

Add `cache_type_k_mixed` / `cache_type_v_mixed` to `llama_memory_params` (in `struct llama_memory_params` in `llama-memory.h`), and populate from `cparams`.

### 6. `src/llama-model.cpp` — Instantiate mixed cache

In `create_memory()`, the existing `default:` branch that creates `llama_kv_cache` (line 2206), add:

```cpp
if (params.cache_type_k_mixed.types != nullptr) {
    kv_cache_mixed_buckets cfg = parse_mixed_spec(params.cache_type_k_mixed.types);
    res = new llama_kv_cache_mixed(
        *this, hparams, cfg, params.type_k,
        !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified,
        cparams.n_ctx_seq, cparams.n_seq_max, 1,
        hparams.n_swa, hparams.swa_type,
        nullptr, filter, nullptr, nullptr);
} else {
    res = new llama_kv_cache(...);  // existing path
}
```

### 7. `src/llama-kv-cache.cpp` / `src/llama-graph.cpp` — Graph building

The attention code needs to handle the mixed cache's merged K/V tensor. When `get_k` returns a merged F16 tensor (built from multiple sub-caches), the attention just sees a normal F16 view — no change needed.

---

## How `get_k` builds the merged view

```cpp
ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) {
    // Create a temporary F16 tensor for the full attention range
    ggml_tensor * merged = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
        n_embd_gqa, n_kv);

    uint32_t head_pos = get_current_head_pos();

    for (uint32_t i = 0; i < n_kv; i++) {
        uint32_t pos = (sinfo_pos0 + i) % kv_size;
        int b = get_bucket_for_pos(pos, head_pos);

        // Copy row from sub_cache[b] into merged
        ggml_tensor * row = ggml_view_1d(ctx,
            sub_caches[b]->get_k(...), n_embd_gqa,
            ggml_row_size(sub_caches[b]->type_k(), n_embd_gqa) * pos);

        ggml_cpy(ctx, row,
            ggml_view_1d(ctx, merged, n_embd_gqa,
                ggml_row_size(GGML_TYPE_F16, n_embd_gqa) * i));
    }

    return merged;
}
```

This creates `n_kv` `ggml_cpy` nodes — one per token in the attention range. For a prompt with 1024 tokens, that's 1024 cpy nodes. This is the main cost: **1024 extra graph nodes per decode step**.

### Optimization: skip when range is within one bucket

If the entire attention range fits in a single bucket (e.g., first 1000 tokens of prompt), `get_k` can just return the sub-cache's tensor view directly — zero overhead.

---

## Migration in `apply()`

After writing a new token to bucket 0:

1. Compute the eviction position for bucket 0→1 boundary: `pos - bucket0_capacity`
2. If that position has valid data in bucket 0, add a `ggml_cpy` from bucket 0 pos to bucket 1 pos
3. Same for bucket 1→2 boundary
4. Actually perform the migration: copy + re-quantize

This adds **2 extra `ggml_cpy` nodes per decode step** (one per bucket boundary crossed).

---

## Memory savings

| Config | Per 256-dim vec | 131K × 4 × 64 layers | vs Q4_0 all |
|--------|----------------|----------------------|-------------|
| Q4_0 all | 72 B | 22 GB | baseline |
| **Q8(1K)+Q4(10K)+Q2(120K)** | **44 B avg** | **13.5 GB** | **-8.5 GB (-39%)** |
| **Q8(512)+Q4(4K)+Q2(126K)** | **38 B avg** | **11.7 GB** | **-10.3 GB (-47%)** |

---

## Speed impact

| Phase | Cost |
|-------|------|
| Normal decode (within bucket 0) | 0 extra |
| Write + migrate (1→2 boundaries) | 2 `ggml_cpy` nodes |
| Attention reading across buckets | N `ggml_cpy` nodes (N = n_kv in range) |

The "reading across buckets" cost can be significant for large prompts: 1024 cpy nodes → maybe ~1-2 ms on GPU. At 300 t/s this is ~30-60% overhead.

**Optimization**: If Q4_0 fits in VRAM (it does for short contexts), the mixed cache is disabled. Only enable when `n_ctx_seq` exceeds a threshold (e.g., 32K).

---

## Implementation order

1. `llama-kv-cache-mixed.h` + `.cpp` — the class
2. `llama-memory.h` — add `kv_cache_mixed_buckets` to params
3. `llama-model.cpp` — instantiate in `create_memory()`
4. `include/llama.h` — add params
5. `common/arg.cpp` — CLI arg
6. `common/common.h` + `.cpp` — pass through
7. `llama-context.cpp` — wire up
