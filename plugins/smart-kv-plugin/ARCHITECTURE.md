# StratumCache — Architecture & Plan

## Overview

Scoring-based tiered KV cache with learned MLP models for llama.cpp.
Two independent decision models replace the old single-score approach:

```
Input Features (17+)
     │
     ├──► QuantRegressor ──► quant tier (1-5) or qtable index (0-11)
     │
     └──► RAMDemoter    ──► GPU vs CPU RAM (binary)
```

---

## Decision 1: Quant Level (QuantRegressor)

**What:** Which quantization type within the GPU VRAM tier?
**Type:** Ordinal regression over the 12-level qtable
**Output:** Continuous score 0-1, mapped to tier 1-5 thresholds

| Tier | Name | Example Type | Bits |
|------|------|-------------|------|
| 1 | Very High | q8_0 / f16 | 8-16 |
| 2 | Quality | q5_1 / q5_0 | 5 |
| 3 | Balanced | q4_0 / q4_k | 4 |
| 4 | Performance | q3_k | 3 |
| 5 | Ultra | q2_k / iq2_s / iq1_s | 2-1.56 |

**Tier → qtable quality band mapping:**

```
Tier 1: f16, q8_0             quality 0.995-1.000  (16-8 bit)
Tier 2: q6_k, q5_k, q5_1, q5_0  quality 0.970-0.990  (6-5 bit)
Tier 3: q4_k, q4_0            quality 0.945-0.960  (4-4.5 bit)
Tier 4: q3_k                  quality 0.930        (3.5 bit)
Tier 5: q2_k, iq2_s, iq1_s    quality 0.780-0.880  (2-1.56 bit)
```

**Note:** Tier 5 spans a wide range (0.780–0.880). The `iq1_s` cliff at 0.780
means the MLP must be confident when assigning tier 5 — borderline chunks
near the tier 4/5 boundary should stay at tier 4 if uncertain.

**Architecture:**
```
Input(18) → Linear(18→16) → ReLU → Linear(16→1) → Sigmoid → score
Params: ~321  (~1.3 KB)
Forward: ~0.5μs
```

The score is thresholded using the existing `smart_kv_assign_tier` mechanism.
Thresholds are adjusted by adaptive gamma under memory pressure.

---

## Decision 2: RAM Demotion (RAMDemoter)

**What:** Should this chunk leave GPU VRAM entirely?
**Type:** Binary classification
**Output:** Probability 0-1 (threshold at 0.5, configurable)

**Architecture:**
```
Input(19) → Linear(19→8) → ReLU → Linear(8→1) → Sigmoid → probability
Params: ~169  (~0.7 KB)
Forward: ~0.3μs
```

**Additional input features** vs QuantRegressor:
- `memory_pressure`: current VRAM usage / capacity (0-1)
  (19th feature; QuantRegressor gets 18 base features without memory_pressure)
- Higher pressure → more aggressive demotion

**Cost weighting:**
- False positive (evicted but needed): 3× penalty (high latency on decode)
- False negative (kept on GPU but unused): 1× penalty (only VRAM waste)

**Label source:**
- During data collection, log: "was this chunk re-accessed within the next N steps?"
- If re-accessed → label 0 (keep on GPU)
- If stale for N steps → label 1 (safe to evict)
- N = 2048 steps (one scoring interval)

---

## Combined Flow

```
For each chunk at scoring interval:

  1. Extract features (17-dim: recency, frequency, attn_ema, query,
     tag_onehot[10], redundancy, chunk_age, pinned, gamma)

  2. RAMDemoter(chunk_features + memory_pressure)
     if probability > threshold → assign to tier 6 (CPU TQ), skip step 3

  3. QuantRegressor(chunk_features)
     score → map to tier 1-5
     enforce min_tier (pinned chunks never demoted below floor)

  4. Enqueue migration if target_tier ≠ current_tier

---

## Promotion Path (Tier 6 → GPU)

When a tier-6 (RAM) chunk is retrieved, it must be promoted back to GPU.

**Trigger:** Any `kv_cache_retrieve` call on a tier-6 slot.
**Action:**
1. Decode from TQ format back to F16 (the normal retrieve path)
2. Clear `slot_tier[slot]` to 0 — this marks the chunk as "no longer evicted"
3. On the next scoring cycle, the chunk's score determines its GPU tier

**Why this works:** After retrieval, the F16 tensor still owns the slot's
position (all tier 6 does is zero the GPU memory). The next write or scoring
pass sees the chunk as tier 0 and assigns a fresh tier 1-5.

**No explicit promotion scoring needed:** The natural recency bump from
the retrieval (last_used_step updated, access_count incremented) guarantees
the chunk scores higher on the next cycle.

**Potential issue:** If a chunk is repeatedly tier-6 → retrieved → tier-1 →
stale → tier-6 → retrieved → ..., this "ping-pong" wastes TQ encode/decode
cycles. Mitigated by migration backoff (skip_counter) and the
FP-heavy RAMDemoter loss weight.
```

---

## Combined vs Separate Models

| Aspect | Single model (old) | Two models (current) |
|--------|-------------------|---------------------|
| Params | 609 | ~562 |
| Forward time | ~1μs | ~1μs |
| RAM demotion accuracy | implicit via threshold | explicit, tunable |
| Memory pressure handling | via adaptive gamma only | direct feature input |
| Training data needs | moderate | moderate (can reuse) |
| Code complexity | simpler | slightly more |
| Calibration | one sigmoid for everything | specialized per task |

**Verdict:** Two models. The RAM demotion decision has a fundamentally different
cost structure (FP expensive, FN cheap) and depends on global memory pressure,
not just per-token importance. A specialized binary classifier handles this better
than a threshold on a general-purpose score.

---

## Input Features (21-dim base)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | recency_log | 0-1 | log(age+1)/16 — normalized time since last access |
| 1 | frequency_log | 0-1 | log(access_count+1)/6 — normalized access frequency |
| 2 | attention_ema | 0-1 | Running EMA of attention mass received |
| 3 | query_score | 0-1 | Query-key similarity to current window (stale at write time) |
| 4-13 | tag_onehot[10] | 0/1 | One-hot encoded chunk tag |
| 14 | redundancy_score | 0-1 | Boilerplate/low-info penalty |
| 15 | chunk_age_ratio | 0-1 | pos_begin / context_length |
| 16 | pinned | 0/1 | Is chunk explicitly pinned? |
| 17 | gamma | 1-4 | Adaptive gamma value |
| 18 | k_norm | 0-~10 | Running mean L2 norm(K) / sqrt(head_dim) — high = likely important |
| 19 | v_norm | 0-~10 | Running mean L2 norm(V) / sqrt(head_dim) |
| 20 | k_variance | 0-~5 | Running variance of K norms across chunk tokens — high = diverse content |

**RAMDemoter only (22nd feature):**
| 21 | memory_pressure | 0-1 | used / capacity |

**Note:** `k_norm` alone is often more predictive than `redundancy_score` or `tag_onehot`
for distinguishing important from boilerplate content. Large K norms correlate with
tokens that produce large attention logits — the model's own implicit importance signal.

**Budget note:** Tag one-hot consumes 10/21 features (~48%). In practice, some tags
(file_path, command) may be rare. Run `tag_usage_counts = Counter(features[:, 4:14].sum(0))`
after data collection — any tag with <1% usage can be dropped to free feature slots.

---

## Training Pipeline

### Data Collection

Done by the plugin during inference (`collect_training_data.py` drives scenarios):

```python
# For each chunk at scoring time:
features = extract_features(chunk_meta)

# Quant label: what tier did the chunk "deserve"?
# Proxy: did the model attend to this chunk later?
if re_attended_within_N_steps:
    quant_label = map(attention_mass → tier)
else:
    quant_label = 0.0  # low tier

# RAM label: was it accessed again?
ram_label = 0 if re_accessed else 1

ls_ring_push(quant_ring, features, quant_label)
ls_ring_push(ram_ring, features + [memory_pressure], ram_label)
```

### Training

```
python train_learned_score.py training_data.bin
```

The training script learns both models from the same dataset:

1. **QuantRegressor**: MSE loss on normalized tier (1-5 → 0.2-1.0)
2. **RAMDemoter**: Weighted BCE loss (FP=3×, FN=1×)

Architecture in training:

```python
class QuantRegressor(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(18, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

class RAMDemoter(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(19, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )
```

### Export

Each model exports to its own binary:
```
quant_weights.bin  (18×16 + 16 + 16×1 + 1 = 321 floats)
ram_weights.bin    (19×8 + 8 + 8×1 + 1 = 169 floats)
```

---

## Benchmark Methodology

Compare perplexity and VRAM across cache policies:

| Policy | Cache Config | VRAM | PPL |
|--------|------------|------|-----|
| F16 (oracle) | `--cache-type-k f16` | high | baseline |
| Q8 uniform | `--cache-type-k q8_0` | medium | +0.X% |
| Q4 uniform | `--cache-type-k q4_0` | low | +X% |
| Mixed bucket | `--cache-type-k-mixed q5_0:2048,q4_0:8192,q2_k:0` | low | +X% |
| Smart heuristic | heuristic scorer | low | +X% |
| Smart learned | learned QuantRegressor | low | +X% |
| Smart + TQ | learned + RAMDemoter | lowest | +X% |

**Measurement:**
1. Start server with profile
2. Send multi-turn conversations (coding scenarios)
3. Collect logprobs from API responses
4. Compute per-token perplexity
5. Measure VRAM from server logs

**Tool:** `benchmark_quality.py --trials 3`

---

## Key Design Principles

1. **RAM demotion is conservative**: False positives (evicting needed data)
   are 3× more expensive than false negatives (keeping unused data on GPU).
   The RAMDemoter should be calibrated with high precision at the cost of recall.

2. **Quant regression is continuous**: Tier boundaries are soft. Under memory
   pressure (high gamma), scores compress and chunks naturally fall into lower tiers.
   No hard boundaries except min_tier (for pinned content).

3. **Training data is perishable**: Session data collected today may not match
   tomorrow's usage patterns. Cross-session weight averaging prevents overfit
   to any single session.

4. **Fallback always works**: If the plugin fails or weights are missing, the
   heuristic scorer (recency + frequency + pin) provides a reasonable default.

---

## File Layout

```
smart-kv-plugin/
├── smart-kv-cache.h        # Core API: chunk meta, weights, inline scorers
├── smart-kv-cache.c        # Quality table, tier mapping, tag defaults
├── learned-score.h         # MLP models (QuantRegressor + RAMDemoter)
├── smart-tq-plugin.cpp     # Combined plugin: scoring + TQ + data collection
├── train_learned_score.py  # PyTorch training (both models)
├── collect_training_data.py # Multi-trial data collector
├── benchmark_quality.py    # Quality benchmark across cache configs
├── test-smart-kv-cache.c   # Unit tests (11 scenarios)
├── test-tq-integration.cpp # TQ pipeline integration test
├── ARCHITECTURE.md         # This file
├── PATCHES.md              # llama.cpp mixed-cache patches
├── README.md               # Quick start
└── SMART_KV_PLAN.md        # Original design doc (legacy)
```

---

## Implementation Status

- [x] Heuristic scorer (recency, frequency, pin, redundancy)
- [x] Tier mapping (1-6, SMART_KV_TIER_COUNT=6)
- [x] Prefill analysis (tag + min_tier from raw text)
- [x] Adaptive gamma (tightens under memory pressure)
- [x] Migration backoff (skip_counter, min_residency)
- [x] Tag weight modifiers (per-tag pin/recency/redundancy multipliers)
- [x] Learned MLP scorer (single model, 17→32→1, legacy)
- [x] Training data collection + export
- [x] PyTorch training pipeline
- [x] Two-model architecture (QuantRegressor 21→16→1 + RAMDemoter 22→8→1)
- [x] K/V norm features (k_norm, v_norm, k_variance)
- [x] Promotion path (tier 6 → GPU on re-access)
- [x] Counterfactual RAM labels (fixup_labels, breaks circularity)
- [x] Deferred re-access tracking in ring buffer export
- [ ] Medium model (26-dim, 32 hidden, ~1.1K params)
- [ ] Large model (36-dim, 64 hidden, ~4.4K params)
- [ ] XL model (52-dim, 128 hidden, ~20K params, confidence head)
- [ ] Random K/V projections for XL content awareness
- [ ] RAMDemoter with memory_pressure feature
- [ ] Weighted BCE loss (FP=3×) for RAM demotion
- [ ] Cross-session weight averaging
- [ ] Online adaptive training (SGD during inference)
- [ ] Attention-feedback labels (real attention weights)
- [ ] Per-layer MLP (separate weights per decoder layer)
- [ ] Per-head tiering (different tiers per GQA head)

---

## Feature Tier Roadmap (Nano → Medium → Large → XL)

Nano is the current implementation (21-dim, 321 params). Larger tiers share the
Nano trunk and add features progressively — Nano weights are the reusable base.

### Medium — 26 dim (+8 over Nano)

Adds K/V full stats, session awareness, and attention entropy:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 18 | k_norm | 0-1 | l2_norm(K) normalized by running max |
| 19 | v_norm | 0-1 | l2_norm(V) normalized by running max |
| 20 | k_variance | 0-1 | variance across chunk's K vectors |
| 21 | v_variance | 0-1 | variance across chunk's V vectors |
| 22 | layer_depth | 0-1 | layer_idx / num_layers |
| 23 | head_idx_norm | 0-1 | head_idx / num_heads |
| 24 | session_token_ratio | 0-1 | tokens_seen / max_context |
| 25 | attn_entropy | 0-1 | entropy of chunk's received attention distribution |

`attn_entropy` is key: low entropy = many queries need the chunk a little
(diffuse), high entropy = few queries need it a lot (spiked). Different
demotion profiles for each.

**Hidden:** 32. **Params:** ~1.1K. **Latency:** ~0.8μs.

### Large — 36 dim (+10 over Medium)

Cross-chunk context and per-head breakdown:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 26 | k_cosine_to_query_mean | -1 to 1 | cosine sim of chunk K to running mean of recent queries |
| 27 | v_cosine_to_recent | -1 to 1 | cosine sim of chunk V to recently recalled V vectors |
| 28 | neighbour_attn_mean | 0-1 | mean attention of ±2 surrounding chunks |
| 29 | neighbour_tier_mean | 0-1 | mean assigned tier of surrounding chunks |
| 30 | head_importance_variance | 0-1 | variance of attn_ema across heads for this chunk |
| 31 | miss_rate_ema | 0-1 | EMA of demotion false positive rate this session |
| 32 | promotion_count | 0-1 | log(times promoted back from tier 6) / 4 |
| 33 | cross_layer_attn_std | 0-1 | std dev of attention received across layers |
| 34 | quant_error_ema | 0-1 | running EMA of observed dequant error for this tier |
| 35 | pinned_neighbour | 0/1 | any pinned chunk within ±4 positions |

`neighbour_tier_mean` and `neighbour_attn_mean` exploit attention locality —
adjacent chunks likely share fate. `miss_rate_ema` closes the calibration
loop: if RAMDemoter has been wrong this session, it learns to be conservative.

**Hidden:** 64. **Params:** ~4.4K. **Latency:** ~1.2μs.

### XL — 52 dim (+16 over Large)

Random projections of K/V subspace for genuine content awareness:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 36-43 | k_proj[8] | -1 to 1 | K @ fixed_random_matrix (8×d_head projection) |
| 44-51 | v_proj[8] | -1 to 1 | V @ fixed_random_matrix (8×d_head projection) |

The random projection matrices are **fixed at init, never trained** — the
Johnson-Lindenstrauss lemma guarantees they preserve inner product geometry.
This gives XL content awareness without learned projection overhead:

```c
// generate once at startup, store in model weights file
void init_random_projections(float* k_proj_mat, float* v_proj_mat,
                              int d_head, int proj_dim, uint64_t seed) {
    srand(seed);
    for (int i = 0; i < d_head * proj_dim; i++) {
        k_proj_mat[i] = gaussian_random() / sqrtf(proj_dim);
        v_proj_mat[i] = gaussian_random() / sqrtf(proj_dim);
    }
}

// at encode time — cheap matmul after mean-pooling K across chunk
void extract_kv_proj(float* out, float* K_mean,
                     float* proj_mat, int d_head, int proj_dim) {
    memset(out, 0, proj_dim * sizeof(float));
    for (int i = 0; i < d_head; i++)
        for (int j = 0; j < proj_dim; j++)
            out[j] += K_mean[i] * proj_mat[i * proj_dim + j];
}
```

XL also gets a **third output head** on the same trunk for confidence:

```
Input(52) → Linear(52→128) → ReLU → Linear(128→64) → ReLU
                                                          │
                        ┌─────────────────────────────────┤
                        ↓                 ↓               ↓
              Linear(64→1)       Linear(64→1)    Linear(64→1)
              quant quality      p_demote        confidence
```

The confidence head gates decisions at inference:

```c
if (confidence < 0.6f) {
    // XL is uncertain — fall back to Medium's decision
    quality_score = lerp(medium_score, xl_score, confidence / 0.6f);
}
```

**Hidden:** 128. **Params:** ~20K. **Latency:** ~2.5μs.

### Architecture Summary

| Tier | Input | Hidden | Params | Latency | When to use |
|------|-------|--------|--------|---------|-------------|
| Nano | 21 | 16 | ~321 | ~0.5μs | Always (current default) |
| Medium | 26 | 32 | ~1.1K | ~0.8μs | >200 samples in ring buffer |
| Large | 36 | 64 | ~4.4K | ~1.2μs | >2000 samples + memory pressure |
| XL | 52 | 128 | ~20K | ~2.5μs | >10000 samples, research only |

### Feature Availability Tiers

Not all features are free — they tier by when computation is possible:

```
Free at write time (always compute, include in all tiers):
  k_norm, v_norm, k_variance, k_proj, v_proj,
  layer_depth, head_idx_norm, session_token_ratio

Cheap at scoring interval (compute per chunk, Medium+):
  attn_entropy, neighbour_attn_mean, neighbour_tier_mean,
  head_importance_variance, cross_layer_attn_std

Requires history (accumulate over session, Large+):
  miss_rate_ema, quant_error_ema, promotion_count,
  k_cosine_to_query_mean, v_cosine_to_recent
```

**Cold start note:** Large/XL features that require history should warm to
0.5 (neutral), not 0, to avoid biasing the model during the first ~100 tokens
before signal accumulates.
