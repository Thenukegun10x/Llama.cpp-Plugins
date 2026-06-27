# Smart KV Cache — Architecture

## Overview

Scoring-based tiered KV cache with learned MLP models for the smart-tq-plugin in llama.cpp. Two independent decision models replace the old heuristic-only approach:

```
Input Features (23-dim)
     │
     ├──► QuantRegressor ──► score (0-1) → mapped to tier 1-5 threshold
     │
     └──► RAMDemoter ──► GPU vs CPU RAM (binary, with memory_pressure)
```

Both models are trained from **heuristic teacher labels** collected during long multi-turn sessions, then exported as float32 weight binaries (~490 floats each).

---

## Decision 1: Quant Level (QuantRegressor)

**What:** Which quantization type within the GPU VRAM tier?
**Type:** Ordinal regression (continuous score → tier thresholds)
**Output:** Score 0-1 → mapped to tier 1-5 via `smart_kv_assign_tier`

| Tier | Name          | TQ Bits | Quality |
|------|---------------|---------|---------|
| 1    | Very High     | 0 (F16) | 1.000   |
| 2    | Quality       | 5       | ~0.994  |
| 3    | Balanced      | 4-5     | ~0.969  |
| 4    | Performance   | 2-4     | ~0.959  |
| 5    | Ultra         | 2       | ~0.941  |
| 6    | Ultra-TQ (RAM)| —       | N/A     |

**Architecture:**
```
Input(23) → Linear(23→16) → ReLU → Linear(16→1) → Sigmoid → score
Params: ~401  (~1.6 KB)
Forward: ~0.5μs
```

The score is thresholded by `smart_kv_assign_tier` using `priority_thresholds` computed from `gamma`:
```c
thresholds = {0.80, 0.55, 0.32, 0.18, 0.08} raised to pow(1/gamma)
```
With gamma=2.0 (balanced): `{0.894, 0.742, 0.566, 0.424, 0.283}`.

---

## Decision 2: RAM Demotion (RAMDemoter)

**What:** Should this chunk leave GPU VRAM for CPU RAM (tier 6)?
**Type:** Binary classification
**Output:** Probability 0-1 (threshold at 0.5, configurable via `--ram-threshold`)

**Architecture:**
```
Input(24) → Linear(24→8) → ReLU → Linear(8→1) → Sigmoid → probability
Params: ~209  (~0.8 KB)
Forward: ~0.3μs
```

**RAMDemoter gets one extra feature** vs QuantRegressor:
- `memory_pressure`: current VRAM usage / capacity (0-1) — 24th feature

**Cost weighting during training:**
- False positive (evicted but needed): **3× penalty** (high latency on CPU→GPU decode)
- False negative (kept on GPU but unused): 1× penalty (only VRAM waste)

**Runtime RAMDemoter path** (when both models are loaded):
1. If `memory_pressure ≤ 0.50` or chunk is pinned → skip RAMDemoter entirely
2. Run `rd_forward(…)` → if `probability ≥ ram_threshold` (default 0.5) → tier 6
3. Otherwise fall through to QuantRegressor for tier 1-5 assignment

---

## Combined Flow

```
For each token via kv_cache_store:

  1. Update chunk metadata: last_used_step, access_count, k/v norm Welford update

  2. score_slot(slot):
     a. If learned model loaded: qr_forward() → q_score
        - If memory_pressure > 0.50 and not pinned: rd_forward() → ram_prob
          → ram_prob ≥ threshold → tier 6, skip step b
        - Graduated offload: if q_score ≤ offload_threshold(pressure) → tier 6
        - smart_kv_assign_tier(q_score, min_tier, thresholds) → tier 1-5
     b. If heuristic fallback: smart_kv_eval() → tier

  3. Enqueue tier change if target_tier ≠ current_tier

  4. kv_cache_store: collect training sample (current slot + 2 cold probes)
     → teacher heuristic labels → push to ring buffer

  5. Every 4096 samples: export rolling dataset to training_data_latest.bin
```

---

## Heuristic Teacher Labels

Training data labels come from the heuristic scorer, NOT from the learned model (no circularity):

```
teacher = smart_kv_eval(chunk, step, max_access, &cfg)

quant_label: 1.0 - (teacher_tier - 1) / 5
  → maps tier 1→1.0 (keep at max quality) through tier 6→0.0 (evict)

ram_label (heuristic):
  bool low_priority    = teacher.priority <= 0.35f
  bool cold            = age > kv_size/16 (2048 steps) || access_count <= 1
  bool stale_rewrite   = total_writes > 1 && access_count <= 1
  bool pressure_cand   = smart_kv_should_offload(…)
  ram_label = !pinned && low_priority && (cold || stale_rewrite || pressure_cand)
```

### Counterfactual Fixup (anti-circularity)

When a chunk is labeled `ram_label=1` (evict) but gets **re-accessed** before the ring buffer rotates out, the label is corrected to 0. This prevents the model from learning to evict chunks that are actually needed:

```
ls_ring_fixup_labels(ring, slot_id, step, kv_size):
  for each sample matching chunk_id AND ram_label > 0.5:
    age = step - sample.last_used_step
    if age < kv_size / 16:    # re-accessed within the "cold" window
      sample.ram_label = 0.0  # teacher was wrong
```

---

## Input Features (23-dim base)

### Core Features (23-dim)

| Index | Feature              | Range   | Description |
|-------|----------------------|---------|-------------|
| 0     | recency_log          | 0-1     | log(age+1)/log(kv_size+1) — normalized time since last access |
| 1     | frequency_log        | 0-1     | log(access_count+1)/log(max_access+1) — normalized access count |
| 2     | attention_ema        | 0-1     | Running EMA of attention mass from K/V norm signal |
| 3     | query_score          | 0-1     | Query-key similarity (stale at write time; uses running norm) |
| 4-13  | tag_onehot[10]       | 0/1     | One-hot encoded chunk tag (system, tool, code, error, …) |
| 14    | redundancy_score     | 0-1     | Boilerplate/low-info penalty from tag+position |
| 15    | chunk_age_ratio      | 0-1     | pos_begin / context_length — early chunks age differently |
| 16    | pinned               | 0/1     | Is chunk explicitly pinned? (system prompts, tool schemas) |
| 17    | gamma                | 1-4     | Adaptive gamma value (drives tier threshold compression) |
| 18    | k_norm               | 0-~10   | Running L2 mean norm(K) / sqrt(head_dim) — high = important |
| 19    | v_norm               | 0-~10   | Running L2 mean norm(V) / sqrt(head_dim) |
| 20    | k_variance           | 0-~5    | Running variance of K norms across chunk tokens — high = diverse |
| 21    | promotion_count_log  | 0-1     | log(promotion_count+1)/4 — times promoted back from tier 6 |
| 22    | recently_promoted    | 0/1     | last_promotion_step within 2048 steps of current step |

### RAMDemoter extra (24th feature)

| Index | Feature          | Range | Description |
|-------|------------------|-------|-------------|
| 23    | memory_pressure   | 0-1   | used / capacity — higher = more aggressive eviction |

### Feature Construction (`ls_features_from_meta`)

```c
ls_features_t ls_features_from_meta(chunk_meta* c, uint64_t step,
                                     uint32_t kv_size, float gamma) {
    uint64_t age = step > c->last_used_step ? step - c->last_used_step : 0;
    feat.recency_log         = logf((float)(age + 1)) / logf((float)(kv_size + 1));
    feat.frequency_log       = logf((float)(c->access_count + 1)) / logf((float)(cached_max_access + 1));
    feat.attention_ema       = c->attention_ema;
    feat.query_score         = c->query_score;
    tag_onehot(…);
    feat.redundancy_score    = c->redundancy;
    feat.chunk_age_ratio     = (float)c->pos_begin / (float)context_length;
    feat.pinned              = c->pinned ? 1.0f : 0.0f;
    feat.gamma               = gamma;
    feat.k_norm              = c->k_norm;
    feat.v_norm              = c->v_norm;
    feat.k_variance          = sqrtf(c->k_variance / (float)(c->n_tokens + 1));
    feat.promotion_count_log = logf((float)(c->promotion_count + 1)) / 4.0f;
    feat.recently_promoted   = (c->promotion_count > 0 &&
                                (step - c->last_promotion_step) < 2048) ? 1.0f : 0.0f;
}
```

### Legacy 21-dim Compatibility

Datasets exported from older builds (before promotion features were added) have 21 features instead of 23. The trainer auto-pads them with zeros for the two missing promotion features:
```python
if n_features == 21:
    X = np.pad(X, ((0,0), (0,2)))  # pad promotion_count_log, recently_promoted
```

---

## Promotion Path (Tier 6 → GPU)

When a tier-6 (RAM) chunk is retrieved, it promotes back to GPU.

**Trigger:** `kv_cache_retrieve` on a tier-6 slot that has TQ-encoded data.

**Promotion steps:**
1. Decode TQ → F16 (normal retrieve path)
2. Increment `promotion_count` (`smart-tq-plugin.cpp:935`)
3. Set `last_promotion_step = ctx->step` (`smart-tq-plugin.cpp:936`)
4. Free TQ CPU buffer
5. The chunk's next scoring pass (after retrieval bumps `last_used_step` and `access_count`) assigns a fresh GPU tier 1-5

**Ping-pong prevention (four mechanisms):**
1. **Promotion counter → feature**: `promotion_count_log` + `recently_promoted` feed into the learned model. A chunk that ping-pongs gets higher promotion features, making eviction LESS likely.
2. **Migration backoff**: `smart_kv_should_skip()` blocks re-migration for `min_residency` steps after any tier change.
3. **Counterfactual fixup**: If a chunk is labeled evict but re-accessed, the label is corrected to 0 in the training ring buffer.
4. **FP=3× loss weight**: The RAMDemoter is penalized 3× for false evictions, making it conservative by default.

---

## Training Data Collection

### Sampling Strategy

Every `kv_cache_store` collects **3 training samples**:

| Sample | Slot | Rate |
|--------|------|------|
| Current token | `slot` | Every store (always occupied) |
| Cold probe 1 | `slot - 256` (forward scan) | Always — probes forward for occupied |
| Cold probe 2 | `slot - 512` (forward scan) | Always — probes forward for occupied |

The cold probe scans **forward (+1)** from the target offset to find the youngest occupied slot, avoiding empty slots in sparsely-filled caches:

```c
int64_t cold = (slot + kv_size + offset) % kv_size;
while ((cold == slot || chunks[cold].access_count == 0) && probes < kv_size/4) {
    cold = (cold + 1) % kv_size;  // scan toward current slot (warmer)
    probes++;
}
```

This guarantees cold samples even when cache fill is low (~5%), and finds younger (warmer) occupied slots that are less likely to be labeled eviction candidates.

### Ring Buffer

- **Capacity:** 131,072 samples (configurable at compile time)
- **Overwrite:** Circular — oldest samples replaced by newest
- **Export:** `training_data_latest.bin` written every 4,096 samples (live monitoring)
- **Session export:** Written on `cache_clear` (server shutdown / checkpoint restore)
- **RAM pos tracking:** Live `ram_pos=X/Y.Z%` in rolling export log line

### Heuristic Teacher (label generation)

```c
// kv_cache_store → collect_training_sample(slot):
teacher = smart_kv_eval(chunk, step, max_access, &cfg)

quant_label = 1.0 - (teacher_tier - 1) / 5
// tier 1 → 1.0 (keep at F16), tier 6 → 0.0 (evict)

ram_label = 0.0
if (!pinned && teacher.priority <= 0.35f && (cold || stale_rewrite || pressure_candidate))
    ram_label = 1.0
```

Where:
- `cold = age > kv_size/16 || access_count <= 1`
- `stale_rewrite = total_writes > 1 && access_count <= 1`
- `pressure_candidate = smart_kv_should_offload(memory_used, memory_capacity, priority)`

### Threshold Selection History

The teacher threshold was tuned iteratively:
- 0.35 → too tight, ~0% positives
- 0.50 → still too tight
- 0.55 → some positives but inconsistent
- 0.60 → better but still low
- 0.566 → matches tier 4 boundary (original `tier >= 4` condition), ~44% at high fill
- 0.35 (current) → combined with warmer cold probes (-256/-512 forward scan) to target 5-20%

---

## Training Pipeline

### Script
```bash
python train_learned_score.py plugins/smart-kv-plugin/training_data_latest.bin plugins/smart-kv-plugin
```

### Dataset Format (binary)

```
[header] 8 bytes:  n_samples (float), n_features (float)
[X]       n * n_features * 4 bytes:  feature matrix (float32)
[y_quant] n * 4 bytes:  quant labels (float32, 0-1)
[y_ram]   n * 4 bytes:  ram labels (float32, 0 or 1)
[mp]      n * 4 bytes:  memory_pressure at collection time (float32, 0-1)
```

### QuantRegressor Training

- **Loss:** MSE on normalized tier (1-5 → 0.2-1.0; tier 6 → 0.0)
- **Architecture:** `Input(23) → Linear(23→16) → ReLU → Linear(16→1) → Sigmoid`
- **Output:** quant_weight.bin (23×16 + 16 + 16×1 + 1 = 401 floats)

### RAMDemoter Training

- **Loss:** Weighted BCE (FP=3×, FN=1×)
- **Architecture:** `Input(24) → Linear(24→8) → ReLU → Linear(8→1) → Sigmoid`
- **Output:** ram_weight.bin (24×8 + 8 + 8×1 + 1 = 209 floats)

### Feature Importance Report

After training, the script prints per-feature L2 weight norms for both models, highlighting promotion features:

```
Feature importance (QuantRegressor):
  recency_log         0.842  <<
  frequency_log       0.531
  attention_ema       0.298
  ...
  promotion_count_log 0.143  <<<< PROMOTION
  recently_promoted   0.097  <<<< PROMOTION
```

Features with `<<` markers are the promotion-tracking features, verifying they contribute non-zero weight.

---

## Memory Pressure Integration

Memory pressure controls **three** independent offload paths:

### 1. Learned RAMDemoter (at memory_pressure > 0.50)
```c
if (lscorer.ram_loaded && pressure > 0.50f) {
    float ram_prob = rd_forward(&feat, pressure);
    if (ram_prob >= g_ram_threshold) return TIER_6;
}
```

### 2. Graduated Offload Threshold (heuristic fallback)
```c
if (pressure > 0.92f) threshold = 0.20f;   // desperate — evict anything cold
else if (pressure > 0.85f) threshold = 0.12f;
else if (pressure > 0.75f) threshold = 0.10f;
else if (pressure > 0.65f) threshold = 0.08f;
else if (pressure > 0.50f) threshold = 0.05f;
// priority ≤ threshold → tier 6
```

### 3. Adaptive Gamma (tier threshold compression)
```c
float gamma = smart_kv_adaptive_gamma(base_gamma, memory_used, memory_capacity);
// Higher gamma → thresholds compress → more chunks fall into lower tiers
```

### Pressure Variance in Training Data

Memory pressure is captured per-sample in the dataset as the `mem_pressure` field. During training, the RAMDemoter receives it as a direct feature, allowing it to learn pressure-dependent eviction behavior. The collector's long-run test profile naturally produces a range of pressure values (mean ~0.166-0.219 observed), so a separate high-pressure collection phase is unnecessary.

---

## Ping-Pong and Cycling Behavior

### ram_pos Cycling

The fraction of positive RAM labels in the ring buffer (`ram_pos`) cycles through each test round:

1. **Start of round** (fresh cache): Cold chunks available → high ram_pos (25-30%)
2. **Mid-round** (cache warming): Cold probe misses empty slots → ram_pos drops
3. **Late round** (cache full, ~22% fill): Stable equilibrium at 44-66%
4. **cache_clear between rounds**: Ring buffer resets → cycle repeats

**Root cause:** Low cache fill (~5%) causes offset-based cold probes (-512, -2048) to hit empty slots ~94% of the time. The fix uses **forward-scan probing** to always find an occupied slot, and **warmer offsets** (-256/-512) to avoid over-labeling.

### Promotion Tracking for Ping-Pong Mitigation

```c
// On tier-6 retrieve:
c->promotion_count++;
c->last_promotion_step = ctx->step;
```

These feed into two features:
- `promotion_count_log`: log-scale count of evict→promote cycles
- `recently_promoted`: binary flag if promoted within last 2048 steps

The model learns: **high promotion_count → don't evict this chunk again**.

---

## Combined vs Separate Models

| Aspect | Heuristic only (fallback) | Learned (two models) |
|--------|--------------------------|---------------------|
| Params | 0 (formula-based) | ~610 |
| Forward time | ~1μs | ~1μs |
| RAM demotion | implicit via tier threshold | explicit, tunable |
| Memory pressure | via adaptive gamma only | direct feature + graduated thresholds |
| Training data | none | needs collection run |
| Robustness | always works | falls back to heuristic if weights missing |

---

## File Layout

```
smart-kv-plugin/
├── smart-kv-cache.h          # Core API: chunk meta, weights, inline scorers
├── smart-kv-cache.c          # Quality table, tier mapping, tag defaults, profiles
├── learned-score.h           # MLP models (QuantRegressor + RAMDemoter), ring buffer
├── smart-tq-plugin.cpp       # Combined plugin: scoring + TQ + data collection + export
├── train_learned_score.py    # PyTorch training (from root dir)
├── collect_training_data.py  # Multi-scenario data collector
├── benchmark_quality.py      # Quality benchmark across cache configs
├── test-smart-kv-cache.c     # Unit tests
├── test-tq-integration.cpp   # TQ pipeline integration test
├── build_plugin.bat          # ROCm clang++ build script
├── ARCHITECTURE.md           # This file
├── PATCHES.md                # llama.cpp mixed-cache patches
├── BUILD.md                  # Build instructions
├── README.md                 # Quick start
└── build/
    └── smart-tq-plugin.dll   # Built plugin binary
```

---

## Build & Deploy

```bash
build_plugin.bat              # ROCm clang++ → build/smart-tq-plugin.dll
# Copy build/smart-tq-plugin.dll → smart-tq-plugin.dll (after server stop)
```

The plugin DLL is loaded at server startup via `llama-server --plugin smart-tq-plugin.dll`. If the DLL is missing or incompatible, the server logs a warning and continues without it.

---

## Implementation Status

- [x] Heuristic scorer (recency, frequency, pin, redundancy, attention, K/V norms)
- [x] Tier mapping (1-6, SMART_KV_TIER_COUNT=6)
- [x] Prefill analysis (tag + min_tier from raw text)
- [x] Adaptive gamma (tightens under memory pressure)
- [x] Migration backoff (skip_counter, min_residency)
- [x] Tag weight modifiers (per-tag pin/recency/redundancy multipliers)
- [x] Learned quant scorer (QuantRegressor: 23→16→1)
- [x] Learned RAM demoter (RAMDemoter: 24→8→1)
- [x] Training data collection + ring buffer + rolling export
- [x] Heuristic teacher labels with counterfactual fixup
- [x] K/V norm features (k_norm, v_norm, k_variance)
- [x] Promotion tracking features (promotion_count_log, recently_promoted)
- [x] Promotion path (tier 6 → GPU on re-access, counter incremented)
- [x] Weighted BCE loss (FP=3×) for RAM demotion
- [x] Feature importance report (per-feature L2 weight norms)
- [x] Legacy 21-dim backward compat (auto-pad with zeros)
- [x] Forward-scan cold probe (guarantees occupied slot, warmer offsets)
- [x] Graduated offload threshold (pressure-dependent)
- [ ] Online adaptive training (SGD during inference)
- [ ] Attention-feedback labels (real attention weights)
- [ ] Per-layer scoring (separate weights per decoder layer)
- [ ] Cross-session weight averaging

---

## Design Principles

1. **Only heuristic teacher labels**: Never label from the learned model itself — avoids circularity. Counterfactual fixup corrects over-eager heuristics.

2. **RAM demotion is conservative**: False positives (evicting needed chunks) cost 3× more than false negatives (wasting VRAM). The loss weight and graduated threshold both push toward precision over recall.

3. **Crash only if broken**: If weights file is missing or mismatched dimension (e.g., old 21-dim weights with new 23-dim model), the plugin falls back to heuristic scoring. Server continues normally.

4. **Offsets age by design**: The -256/-512 cold offsets with forward scan find warmer-but-not-current slots. After ~256 tokens, the recency penalty drops priority enough to test the boundary.

5. **Promotion tracking breaks ping-pong**: Each evict→promote cycle increments a counter that feeds back as model features. Chunks that have been ping-ponged become harder to evict.
