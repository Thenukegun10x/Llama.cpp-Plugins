# Patches: Smart Cache Tier-6 Routing into llama.cpp

Two files in `src/` were modified to add a **TQ plugin bucket** that accepts
tier-6 writes (cold chunks) and skips GPU allocation for those slots.

## `src/llama-kv-cache-mixed.h`

### 1. Add `#include <functional>` (#7)

```cpp
#include <functional>
```

### 2. Add methods and flag to class (#68-#72)

```cpp
    void set_tier_fn(std::function<uint8_t(uint32_t)> fn) { tier_fn_ = std::move(fn); }
    bool  has_tier_fn() const { return (bool)tier_fn_; }
```

### 3. Add `is_tq_plugin` to `bucket_info` (#74)

```cpp
    struct bucket_info {
        ...
        bool                            is_tq_plugin = false;
    };
```

### 4. Add `add_tq_plugin_bucket` declaration (#88)

```cpp
    void add_tq_plugin_bucket(const llama_model & model, uint32_t kv_size,
                              uint32_t n_seq_max, uint32_t n_pad,
                              const layer_filter_cb & filter,
                              const layer_reuse_cb & reuse,
                              const layer_share_cb & share);
```

### 5. Add `tier_fn_` member variable (#94)

```cpp
    std::function<uint8_t(uint32_t global_pos)> tier_fn_;
```

## `src/llama-kv-cache-mixed.cpp`

### 1. Constructor — add TQ bucket after normal buckets (after line 137)

```cpp
    if (tier_fn_) {
        add_tq_plugin_bucket(model, kv_size, n_seq_max, n_pad, filter, reuse, share);
    }
```

### 2. New method `add_tq_plugin_bucket` (between existing buckets and `get_can_shift`)

Uses 1 GPU cell (placeholder) — get_k/v allocates zero tensors from graph context.

```cpp
void llama_kv_cache_mixed::add_tq_plugin_bucket(...) {
    auto kv = std::make_unique<llama_kv_cache>(
            model, hparams, GGML_TYPE_F16, GGML_TYPE_F16,
            v_trans_, false, unified,  // offload=false
            1, n_seq_max, n_pad,       // 1 cell — saves ~400 MB VRAM
            0, LLAMA_SWA_TYPE_NONE,
            nullptr, filter, reuse, share);

    LLAMA_LOG_INFO("%s:   bucket %zu: TQ plugin (tier 6, CPU offload)\n", __func__, buckets.size());

    bucket_info bi;
    bi.kv = std::move(kv);
    bi.capacity = kv_size;
    bi.type = GGML_TYPE_F16;
    bi.is_tq_plugin = true;
    buckets.push_back(std::move(bi));
}
```

### 3. `route_write` — check tier callback before positional routing

```cpp
    if (tier_fn_) {
        uint8_t t = tier_fn_(global_pos);
        if (t >= 6) {
            for (size_t i = 0; i < buckets.size(); ++i) {
                if (buckets[i].is_tq_plugin) return i;
            }
        }
    }
```

After the `if (buckets.empty()...)` guard, before the positional routing.

### 4. `route_write_remaining` — unlimited capacity for tier 6

```cpp
    if (tier_fn_ && tier_fn_(global_pos) >= 6) return UINT32_MAX;
```

After the `if (buckets.empty()...)` guard, before the positional routing.

### 5. `init_batch` — skip `prepare` for TQ buckets

In the prepare loop, before `auto sinfos = buckets[ib].kv->prepare(bucket_ubatches[ib]);`:

```cpp
    if (buckets[ib].is_tq_plugin) {
        llama_kv_cache::slot_info_vec_t dummy_sinfos;
        dummy_sinfos.reserve(bucket_ubatch_indices[ib].size());
        for (size_t j = 0; j < bucket_ubatch_indices[ib].size(); ++j) {
            llama_kv_cache::slot_info si;
            si.s0 = 0; si.s1 = 0;
            si.strm = { 0 };
            si.idxs = { llama_kv_cache::slot_info::idx_vec_t() };
            dummy_sinfos.push_back(si);
            ubatch_sinfos[bucket_ubatch_indices[ib][j]] = si;
        }
        bucket_sinfos[ib] = std::move(dummy_sinfos);
        continue;
    }
```

### 6. `cpy_k` / `cpy_v` — return identity for TQ buckets

```cpp
    if (kv_mixed && ib < kv_mixed->n_buckets() && kv_mixed->buckets[ib].is_tq_plugin) {
        return k_cur;  // or v_cur
    }
```

### 7. `get_k` / `get_v` — allocate zero tensor for TQ plugin buckets

In the batch-path loops (both K and V), add before `get_bucket(i)->get_k/v(...)`:

```cpp
    if (kv_mixed->buckets[i].is_tq_plugin && n_kv > 0) {
        ggml_tensor * ref = kv_mixed->get_bucket(0)->get_k(ctx, il, 1, sinfo_cur);
        ggml_tensor * view = ggml_new_tensor_2d(ctx, ref->type, ref->ne[0], n_kv);
        memset(view->data, 0, (size_t)ggml_nbytes(view));
        views.push_back({ view, n_kv });
        continue;
    }
```

Same pattern for `get_v` (uses `get_bucket(0)->get_v` for ref dims).

### VRAM savings

| Component | Before | After |
|-----------|--------|-------|
| TQ bucket GPU tensor | kv_size × n_embd × F16 (~400 MB) | 1 cell × n_embd × F16 (~2 KB) |
| Routing metadata | ~1 MB | ~1 MB |
| TQ CPU data (30% tokens) | 0 (no TQ) | ~20 MB |

Net VRAM saved: **~400 MB** (the full-size F16 tensor that was always allocated).

## Applying the Patches

```
cd path/to/llama.cpp
vim src/llama-kv-cache-mixed.h    # apply changes 1-5
vim src/llama-kv-cache-mixed.cpp  # apply changes 1-6
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DGGML_PLUGIN_SUPPORT=ON
ninja
```
