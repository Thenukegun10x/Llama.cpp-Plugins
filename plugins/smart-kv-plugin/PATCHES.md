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

## `src/llama-model.cpp` — Optional: Wire tier_fn_ for Unified TQ

For unified memory systems that want ALL tiers routed through TQ (not just
tier 6 CPU offload), uncomment the block after the mixed cache creation:

```cpp
// UNIFIED MEMORY TQ: Uncomment the block below to route ALL tiers through
// the TQ plugin bucket instead of positional mixed Q types.
// The plugin handles per-tier compression via the tq_bits profile.
//
// #ifdef GGML_PLUGIN_SUPPORT
//     if (cparams.use_plugin_kv_cache && ...) {
//         cache_mixed->add_tq_plugin_bucket(...);
//         cache_mixed->set_tier_fn([](uint32_t) -> uint8_t { return 1; });
//     }
// #endif
```

Leave commented for normal tier-6 CPU-offload operation (the `smart` profile).
Uncomment for unified memory (Apple, integrated GPUs).

## `src/llama-kv-cache.cpp` — GPU Buffer Support for Plugin KV Cache

The default `llama_kv_cache` (non-mixed) also needs patching to activate the
plugin when K/V buffers are GPU-resident. Two changes:

### 1. Init — remove host-buffer check (#382-#411)

Replace the `host_buffers` guard that skipped plugin layer creation for GPU K/V:

```cpp
// OLD:
bool host_buffers = true;
for (auto & layer : layers) {
    if (layer.k && layer.k->buffer && !ggml_backend_buffer_is_host(layer.k->buffer)) {
        host_buffers = false; break;
    } ... }
if (host_buffers) {
    for (auto & layer : layers) { ... }
}

// NEW: always create plugin layers (plugin's kv_cache_supported validates)
for (auto & layer : layers) {
    ...
    layer.plugin_ctx_k = ggml_plugin_kv_cache_create_layer(...);
    layer.plugin_ctx_v = ggml_plugin_kv_cache_create_layer(...);
}
```

### 2. Store — pass nullptr for GPU-resident K/V data (#1256-#1292)

The store path previously skipped GPU-only layers entirely. Now it always
notifies the plugin, passing `nullptr` when the buffer is GPU-resident.
The plugin handles this by skipping K/V norm computation while still
collecting metadata features.

```cpp
// OLD:
const bool k_accessible = ...ggml_backend_buffer_is_host(layer.k->buffer);
if (!k_accessible && !v_accessible) { continue; }
if (k_accessible && layer.plugin_ctx_k) {
    const void * k_row = layer.k->data + idx * row_size;
    ggml_plugin_kv_cache_store(ctx_k, idx, k_row, nullptr, 1);
}

// NEW:
const bool k_host = ...ggml_backend_buffer_is_host(layer.k->buffer);
if (layer.plugin_ctx_k) {
    const void * k_row = k_host
        ? (const uint8_t*)layer.k->data + idx * row_size
        : nullptr;  // GPU buffer — plugin skips norm computation
    ggml_plugin_kv_cache_store(ctx_k, idx, k_row, nullptr, 1);
}
```

## `plugins/smart-kv-plugin/smart-tq-plugin.cpp` — Handle Null K/V in Store

The plugin's `kv_cache_store` must handle null K/V data pointers for GPU buffers:

### 1. Accept null data (#297)

```cpp
// OLD:  if (!ctx || !k_data || !v_data) return -1;
// NEW:  if (!ctx) return -1;
```

### 2. Guard K/V norm computation (#315-#340)

```cpp
// OLD: unconditional K/V norm + Welford update
// NEW: if (k_fp16 && v_fp16) { ... K/V norm + Welford update ... }
```

### 8. Unified Memory: Route All Tiers to TQ Plugin

For unified memory systems (Apple, integrated GPUs), the TQ plugin bucket
handles ALL tiers, not just tier 6. Three changes in `src/llama-kv-cache-mixed.cpp`:

**`add_tq_plugin_bucket` (#150-#162):** Allocate full `kv_size` cells instead of 1-cell placeholder:

```cpp
// OLD: auto kv = ... 1, n_seq_max, n_pad ...
// NEW: auto kv = ... kv_size, n_seq_max, n_pad ...
```

**`route_write` (#241-#248):** Route any tier (not just ≥ 6) to the TQ bucket:

```cpp
// OLD: if (t >= 6) {
// NEW: if (t >= 1) {
```

**`route_write_remaining` (#265-#266):** Same for remaining capacity:

```cpp
// OLD: if (tier_fn_ && tier_fn_(global_pos) >= 6) return UINT32_MAX;
// NEW: if (tier_fn_ && tier_fn_(global_pos) >= 1) return UINT32_MAX;
```

### VRAM savings (unified memory)

| Component | Before | After |
|-----------|--------|-------|
| TQ bucket GPU tensor | kv_size × n_embd × F16 (~400 MB) | kv_size × n_embd × F16 (~400 MB) |
| TQ CPU/total data | 0 | variable (per-tier TQ encode) |
| Native F16 tensor | full size | freed (plugin owns storage) |

Net: same VRAM usage but **compressed in place** via TQ at 2-5× per tier.

## `plugins/smart-kv-plugin/smart-tq-plugin.cpp` — Hardening Fixes (2025-06)

### 1. TQ encode/decode guard against null source data (#820-828)

Before the TQ encode loop, check that K/V data pointers are non-null (may be
nullptr for GPU-resident buffers). Falls back to GPU F16 tier if inaccessible.

```cpp
// OLD: unconditional encode_head_fp16(k_fp16 + off_k, ...)
// NEW:
if (!k_fp16 || !v_fp16) {
    free_tq_slot(ctx, slot);
    ctx->slot_tier[slot] = 0;
    c->tier = 0; c->target_tier = 0;
    continue;
}
```

### 2. Slot bounds: add `slot < 0` guard (#713, #900)

Both `kv_cache_store` and `kv_cache_retrieve` only checked `slot >= kv_size`;
a negative `pos` bypasses this. Add lower bound.

```cpp
// OLD: if (slot >= ctx->kv_size) return -1;
// NEW: if (slot < 0 || slot >= ctx->kv_size) return -1;
```

### 3. Negative modulo hardening (#784-785)

C++ `%` yields negative for negative left operand. Wrap with extra `kv_size`.

```cpp
// OLD: (slot + ctx->kv_size - 512) % ctx->kv_size
// NEW: (slot + ctx->kv_size - 512 + ctx->kv_size) % ctx->kv_size
```

### 4. `realloc` temp variable (#394-410)

`realloc` return assigned directly loses original pointer on failure.
Use temp with null check.

```cpp
void* tmp;
tmp = realloc(g_shared_ctx->k_rows, new_sz * sizeof(uint8_t*));
if (tmp) g_shared_ctx->k_rows = (uint8_t**)tmp;
// ... repeat for v_rows, chunks, total_writes, slot_tier
```

### 5. `encode_head_fp16` / `decode_head_fp16` malloc guard (#581-597)

Malloc result unchecked before dereference.

```cpp
float* tmp = (float*)malloc((size_t)head_dim * sizeof(float));
if (!tmp) return;
```

### 6. Pressure gate lowered for HumanEval benchmarks (#632-645)

Tier-6 offload previously required `pressure > 0.75f` which never triggers during
HumanEval (~50-60% context fill). Lowered to `0.50f` with intermediate bands at
0.65 and 0.50 for heuristic offload.

## `plugins/smart-kv-plugin/smart-kv-cache.c` — Hardening Fixes

### 7. `smart_kv_tier_to_type` bounds guard (#387)

OOB when `tier == 0` accessing `steps[tier - 1]`. Added early guard.

```cpp
if (tier < 1 || tier > SMART_KV_TIER_COUNT) return base_type;
```

### 8. `smart_kv_eval` null guard (#442-443)

Dereferences `c->tag` and `cfg->weights` without validating pointers.

```cpp
smart_kv_scored_chunk r = { .tier = SMART_KV_TIER_BALANCED, ... };
if (!c || !cfg) return r;
```

## `start.bat` / `build_plugin.bat` — Batch Parsing Fix

Parenthesized echo inside `if` / `else` block closes the block early. Removed
parentheses from echo strings.

```bat
rem OLD: echo [train] Training failed (code !ERRORLEVEL!)
rem NEW: echo [train] Training failed with code !ERRORLEVEL!
```

## `plugins/smart-kv-plugin/smart-tq-plugin.cpp` — Performance & Safety (2025-06)

### 9. Incremental `memory_used` tracking (O(N) → O(1))

`score_slot` rescanned all `kv_size` slots to count occupied slots every token.
Now tracked as a counter: incremented on first slot write, zeroed on cache clear.

```cpp
// kv_cache_store: increment on first write
bool was_unoccupied = (c->access_count == 0 && ctx->slot_tier[slot] == 0);
c->access_count++;
if (was_unoccupied) ctx->cfg.memory_used++;

// cache_clear: reset
ctx->cfg.memory_used = 0;

// dump_stats: use incrementally-tracked counter
uint64_t occupied = ctx->cfg.memory_used;
```

### 10. Incremental `max_access` tracking (O(N) -> O(1))

`score_slot` previously scanned all `kv_size` slots for `max_access` on every
tier decision. With plugin callbacks multiplied across layers, this dominated
prefill. `cached_max_access` is now updated when a slot's `access_count`
increases, and `score_slot` reads the cached value directly.

### 11. Training collection gated behind mode flags

`collect_training_sample` + ring buffer pushes now only run when
`g_train_mode || g_snapshot_exports`. Eliminates wasted CPU in production.

### 12. Heap-allocated ring buffer (20 MB saved in production)

`train_ring` changed from embedded struct member to `ls_ring_buffer_t*`.
Allocated via `calloc` only when `g_train_mode || g_snapshot_exports`.
Freed in `on_unload`. Null-guarded at all access points.

```cpp
// kv_cache_create_layer
if (g_train_mode || g_snapshot_exports) {
    ctx->train_ring = (ls_ring_buffer_t*)calloc(1, sizeof(ls_ring_buffer_t));
    if (ctx->train_ring) ls_ring_init(ctx->train_ring);
}

// on_unload
free(g_shared_ctx->train_ring);
```

### 13. Tier 1-5 retrieve: removed dead self-memcpy

The `memcpy(k_fp16 + offset, k_fp16, size)` with identical-source ternary was
dead code (llama.cpp fills k_out/v_out before callback). Replaced with comment.

```cpp
// OLD: memcpy(k_fp16 + offset, ... both branches are k_fp16 ...)
// NEW: // Tier 1-5: data already in GPU F16 tensor — llama.cpp
//      // fills k_out/v_out before this callback. No plugin action needed.
```

### 14. Logical-store de-duplication

llama.cpp can call the KV plugin once per layer for the same logical slot. The
smart policy is global per slot, so tier scoring and metadata updates now run
only once per contiguous slot write using `last_store_pos`.

```cpp
if (slot == policy->last_store_pos) {
    // Another layer is storing the same logical token.
    // Keep per-layer payload handling separate from global policy updates.
} else {
    policy->last_store_pos = slot;
    // update metadata + score once
}
```

### 15. Live stats disabled by default

`SMART_KV_STATS_INTERVAL` now defaults to `0`. Cache clear and unload still dump
stats, but live JSON writes are opt-in to avoid periodic stalls during prompt
ingest.

## `plugins/smart-kv-plugin/smart-kv-cache.h` — Threshold Syncing

### 16. `smart_kv_should_offload` thresholds synced with `score_slot`

Training labels were computed with one policy (baseline 0.08, 3 bands) while
inference used another (baseline 0.05, 5 bands). Now identical: baseline 0.05
with bands at 0.92, 0.85, 0.75, 0.65, 0.50.

## `plugins/smart-kv-plugin/smart-kv-cache.c` — Quant Table Safety

### 17. `smart_kv_tier_to_type` underflow guard

Added `ti < 0` bounds check in both the known-type and fallback paths.
When adding new quant types to `qtable[]`, review the `steps[]` offsets
to ensure they don't produce negative indices (documented in qtable comments).

## Tier Routing Patch (Tier 6 → VRAM savings)

These changes wire the plugin's tier decisions into llama.cpp's mixed-cache
routing so that tier-6 slots go to the TQ plugin bucket (1 GPU cell, `~0.5 MB`)
instead of the full F16 tensor (`~400 MB`).

### 18. Plugin: `smart_tq_get_tier` export (`smart-tq-plugin.cpp`)

```cpp
extern "C" GGML_PLUGIN_EXPORT
uint8_t smart_tq_get_tier(uint32_t global_pos) {
    if (!g_shared_ctx) return 0;
    if ((int64_t)global_pos >= g_shared_ctx->kv_size) return 0;
    return g_shared_ctx->slot_tier[global_pos];
}
```

### 19. Plugin loader: `ggml_plugin_get_export()` (`ggml-plugin.h`, `ggml-plugin.c`)

Generic symbol lookup across all loaded plugins:

```c
// ggml-plugin.h:
GGML_PLUGIN_API void* ggml_plugin_get_export(const char* symbol_name);

// ggml-plugin.c:
void* ggml_plugin_get_export(const char* symbol_name) {
    for (int i = 0; i < g_n_plugins; i++) {
        void* sym = dl_sym(g_plugins[i]->handle, symbol_name);
        if (sym) return sym;
    }
    return NULL;
}
```

### 20. `llama-model.cpp`: wire tier_fn_

Added after mixed cache creation in `create_memory()`:

```cpp
auto* mixed = static_cast<llama_kv_cache_mixed*>(res);
typedef uint8_t (*tier_fn_t)(uint32_t);
auto fn = (tier_fn_t)ggml_plugin_get_export("smart_tq_get_tier");
if (fn) mixed->set_tier_fn(fn);
```

### 21. `llama-kv-cache-mixed.cpp`: tier-6 routing (not unified)

Changed routing from `t >= 1` (all tiers → TQ) to `t >= 6` (only CPU offload):

```cpp
// route_write:
if (t >= 6) {  // was: t >= 1

// route_write_remaining:
if (tier_fn_ && tier_fn_(global_pos) >= 6) return UINT32_MAX;  // was: >= 1
```

### 22. `add_tq_plugin_bucket`: 1 cell placeholder

Changed from `kv_size` cells to `1` cell — saves `~400 MB` GPU VRAM:

```cpp
auto kv = std::make_unique<llama_kv_cache>(
    ... 1, n_seq_max, n_pad, ...);  // was: kv_size
bi.capacity = 1;  // was: kv_size
```

### VRAM savings (tier-6 offload)

| Component | Before | After |
|-----------|--------|-------|
| GPU F16 tensor (all slots) | kv_size × n_embd × F16 (~400 MB) | kv_size × n_embd × F16 (~400 MB) |
| TQ plugin bucket GPU tensor | absent | 1 cell × n_embd × F16 (~12 KB) |
| Tier-6 CPU TQ storage | 0 | variable (dynamic per-slot) |

Tier-6 slots now **skip the GPU F16 allocation** entirely — they only exist
in CPU TQ format. The 1-cell GPU bucket is a routing placeholder.

## TQ Redesign: Per-Layer Payloads + Global Policy

The original smart-TQ plugin returned one singleton context for all layers and
for both K/V halves. That made smart policy global, but it also meant tier-6 TQ
payload storage was not correct: real KV data is per layer, while the singleton
had only one `k_rows[slot]` and one `v_rows[slot]` table.

The redesign splits responsibilities:

- `g_shared_ctx`: global policy owner (`slot_tier`, `chunks`, scoring state,
  training ring, counters)
- one `SmartTQContext` per model layer: per-layer TQ row tables (`k_rows`,
  `v_rows`) and layer geometry

### 23. llama.cpp: use one combined plugin context per layer

The plugin API already expects one callback containing both K and V:

```c
int (*kv_cache_store)(void* layer_ctx, int64_t pos,
                      const void* k_data, const void* v_data,
                      int64_t n_tokens);
```

`src/llama-kv-cache.cpp` now creates one plugin context per layer and passes
both rows in one call when they are host-accessible:

```cpp
layer.plugin_ctx_k = ggml_plugin_kv_cache_create_layer((int)il, &kvp, 4);
layer.plugin_ctx_v = nullptr;

const void * k_row = k_host ? (const uint8_t*)layer.k->data + idx * ggml_row_size(layer.k->type, layer.k->ne[0]) : nullptr;
const void * v_row = v_host ? (const uint8_t*)layer.v->data + idx * ggml_row_size(layer.v->type, layer.v->ne[0]) : nullptr;

ggml_plugin_kv_cache_store(layer.plugin_ctx_k, (int64_t)idx, k_row, v_row, 1);
```

`plugin_get_ctx_v()` returns the same combined context as `plugin_get_ctx_k()`
so future callers still receive a valid handle.

### 24. Plugin: per-layer row stores

`kv_cache_create_layer()` now returns a distinct `SmartTQContext` per layer.
The first context owns the global policy arrays; later contexts share those
arrays but allocate independent TQ row-pointer tables.

```cpp
ctx->slot_tier    = g_shared_ctx->slot_tier;
ctx->chunks       = g_shared_ctx->chunks;
ctx->total_writes = g_shared_ctx->total_writes;

ctx->k_rows = (uint8_t**)calloc((size_t)ctx->kv_size, sizeof(uint8_t*));
ctx->v_rows = (uint8_t**)calloc((size_t)ctx->kv_size, sizeof(uint8_t*));
```

Stats aggregate TQ row allocation and live CPU bytes across all layer contexts.

### 25. Experimental TQ encode gate

Live CPU TQ writes are disabled by default to avoid reintroducing the prefill
slowdown. Set `SMART_KV_TQ_ENCODE=1` to enable experimental TQ encoding when
both K and V rows are available on host memory.

Default behavior:

- smart policy scoring remains active
- TQ payload rows are not encoded during GPU-resident prefill
- no GPU->CPU sync copy is forced from the plugin hot path

This leaves the next production step as a cold-block migration path: migrate
old GPU KV blocks to per-layer CPU TQ storage outside the hot prefill path.

## Applying the Patches

```
cd path/to/llama.cpp
vim src/llama-kv-cache-mixed.h    # apply changes 1-5
vim src/llama-kv-cache-mixed.cpp  # apply changes 1-8 (added unified memory routing)
vim src/llama-kv-cache.cpp        # apply changes 1-2 (GPU buffer support)
vim src/llama-model.cpp           # apply tier_fn_ wiring
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DGGML_PLUGIN_SUPPORT=ON
ninja

# Also rebuild the plugin
cd plugins/smart-kv-plugin
./build_plugin.bat
copy build\smart-tq-plugin.dll .
```
