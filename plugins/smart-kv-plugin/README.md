# Smart KV Cache Plugin

Scoring-based KV cache eviction and tier assignment for llama.cpp.

## Files

| File | Purpose |
|------|---------|
| `smart-kv-cache.h` | API: chunk meta, weights, config, inline scorers |
| `smart-kv-cache.c` | Implementation: quality table, tier mapping, tag defaults |
| `test-smart-kv-cache.c` | Standalone test with fake cache dumps |
| `SMART_KV_PLAN.md` | Full design doc from the repo root |

## Scoring Formula

```
S = w_r*R + w_a*A + w_f*F + w_q*Q + w_p*P - w_d*D
```

R = exp(-delta/tau_r) recency decay, A = attention EMA, F = log1p frequency,
Q = query similarity, P = pin boost, D = redundancy penalty.

## Finer Controls

### Per-Chunk Fields

| Field | Type | Description |
|-------|------|-------------|
| `min_tier` | uint8_t | Never demote below this tier (1-5, 0 = no restriction). Errors pinned to tier 1 can't slip to Q2_K under pressure. |
| `tag` | uint8_t | `smart_kv_tag` enum — system, tool_schema, code, error, boilerplate, etc. Triggers heuristic weight modifiers. |
| `skip_counter` | uint32_t | Migration backoff: when set > 0, the chunk is skipped for N scoring cycles. Doubles on no-op to prevent thrash. |
| `anchor_score` | float | 0-1. Unpinned chunks with high anchor (file paths, errors) still get a partial pin boost. |
| `redundancy_score` | float | 0-1. Boilerplate, repeated logs, old assistant prose get penalized here. |

### Tag Weight Modifiers

Each tag has per-component multipliers (initialized in `smart_kv_init_tag_mods()`):

| Tag | recency | frequency | pin | redundancy |
|-----|---------|-----------|-----|------------|
| system | 1.3x | 1.0x | 1.5x | 1.0x |
| tool_schema | 1.0x | 1.5x | 1.3x | 1.0x |
| code | 1.0x | 1.5x | 1.0x | 1.0x |
| error | 1.5x | 1.0x | 1.2x | 1.0x |
| boilerplate | 1.0x | 1.0x | 1.0x | 2.0x |
| assistant | 1.0x | 1.0x | 1.0x | 1.5x |
| file_path | 1.3x | 1.0x | 1.0x | 1.0x |
| command | 1.2x | 1.0x | 1.0x | 1.0x |

These are multipliers on the base score component before the global weight is applied. `smart_kv_init_tag_mods()` sets these; override any with `w->tag_mod_*[tag] = val`.

### Adaptive Gamma

Under memory pressure (used/capacity > 85%), gamma tightens from 2.0 to 1.4, making the tier mapping less aggressive. At >70% it goes to 1.7. This prevents mass evictions when the cache is full.

### Migration Backoff

`skip_counter` on each chunk: incremented on each no-op migration decision, decremented each cycle when idle. Prevents the same chunk from being promoted/demoted repeatedly. Combined with `min_residency` (min tokens between migrations of the same chunk), this eliminates thrash.

### Per-Layer Weight Profiles

Not hardcoded — pass a different `smart_kv_config` per layer. The config holds one `smart_kv_weights` struct; call `smart_kv_eval()` with the appropriate config for each layer. A front-end can store an array `smart_kv_config[il]` and index by layer.

### Step Offsets Per Tier

```c
static const int steps[SMART_KV_TIER_COUNT] = { 0, 1, 3, 5, 7 };
```

Tier 1 (Very High): 0 steps down from base. Tier 5 (Ultra): 7 steps down (capped at table end). Edit this array to change the aggressiveness gradient.

## Adding FP8 / NVFP4 / TQ Types

Add entries to the `qtable` in `smart-kv-cache.c` at the appropriate quality position. The tier step offsets will automatically select them. Example for FP8 (E4M3):

```c
{ ?, "fp8", 0.992f },  // between q8_0 and q6_k
```

Set `base_type` to the new GGML_TYPE enum value in config, and the tier mapping walks down from there. No other code changes needed.

## Running the Test

```sh
cc -o test-smart-kv-cache test-smart-kv-cache.c smart-kv-cache.c -lm
./test-smart-kv-cache
```
