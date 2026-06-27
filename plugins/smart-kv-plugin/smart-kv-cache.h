#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SMART_KV_TIER_COUNT    6
#define SMART_KV_MAX_TAGS      16
#define SMART_KV_MAX_TIER_TYPES 8  // TQ bit widths possible: 2,3,4,5,6, raw F16, Q4_0, Q8_0

typedef enum {
    SMART_KV_TIER_VERY_HIGH    = 1,
    SMART_KV_TIER_QUALITY      = 2,
    SMART_KV_TIER_BALANCED     = 3,
    SMART_KV_TIER_PERFORMANCE  = 4,
    SMART_KV_TIER_ULTRA        = 5,
    SMART_KV_TIER_ULTRA_TQ     = 6,  // CPU-offload via TurboQuant plugin
} smart_kv_tier;

// Built-in chunk tags for heuristic scoring
typedef enum {
    SMART_KV_TAG_DEFAULT      = 0,
    SMART_KV_TAG_SYSTEM       = 1,  // system prompt
    SMART_KV_TAG_TOOL_SCHEMA  = 2,  // tool/function schemas
    SMART_KV_TAG_CODE         = 3,  // code blocks
    SMART_KV_TAG_ERROR        = 4,  // compiler/test errors
    SMART_KV_TAG_USER_INPUT   = 5,  // user messages
    SMART_KV_TAG_ASSISTANT    = 6,  // assistant responses
    SMART_KV_TAG_BOILERPLATE  = 7,  // logs, repeated prose
    SMART_KV_TAG_FILE_PATH    = 8,  // file paths
    SMART_KV_TAG_COMMAND      = 9,  // CLI commands
} smart_kv_tag;

typedef struct {
    uint32_t chunk_id;
    uint32_t pos_begin;
    uint32_t pos_end;
    uint32_t n_tokens;

    uint8_t  tier;
    uint8_t  target_tier;
    uint8_t  min_tier;      // never demote below this tier (1-5, 0 = no restriction)
    uint8_t  tag;           // smart_kv_tag for heuristic scoring
    uint32_t skip_counter;  // migration backoff counter
    bool     pinned;
    bool     dirty;

    float score;
    float recency_score;
    float attention_score;
    float frequency_score;
    float query_score;
    float anchor_score;
    float redundancy_score;

    uint64_t last_used_step;
    uint32_t access_count;
    float    attention_ema;

    // Promotion tracking — prevents tier-6 ping-pong
    uint32_t promotion_count;      // times promoted back from tier-6
    uint64_t last_promotion_step;  // step of most recent tier-6 promotion

    // K/V tensor statistics — cheapest predictive features
    float k_norm;          // running mean L2 norm(K) / sqrt(head_dim)
    float v_norm;          // running mean L2 norm(V) / sqrt(head_dim)
    float k_variance;      // running variance of K norms across tokens in chunk
} smart_kv_chunk_meta;

typedef struct {
    float w_recency;
    float w_attention;
    float w_frequency;
    float w_query;
    float w_pin;
    float w_redundancy;
    float tau_r;
    float lambda_a;
    float gamma;
    float priority_thresholds[SMART_KV_TIER_COUNT];
    float tag_mod_recency[SMART_KV_MAX_TAGS];     // per-tag multiplier on recency
    float tag_mod_frequency[SMART_KV_MAX_TAGS];   // per-tag multiplier on frequency
    float tag_mod_pin[SMART_KV_MAX_TAGS];         // per-tag multiplier on pin
    float tag_mod_redundancy[SMART_KV_MAX_TAGS];  // per-tag multiplier on redundancy
} smart_kv_weights;

typedef struct {
    smart_kv_weights weights;
    uint32_t         score_interval;
    uint32_t         migrate_max;
    int              base_type;
    uint32_t         memory_capacity;  // total KV cells available
    uint32_t         memory_used;      // currently used KV cells
} smart_kv_config;

typedef struct {
    smart_kv_tier tier;
    int           target_type;
    float         priority;
} smart_kv_scored_chunk;

// ── Unified memory TQ profiles ───────────────────────────────────────
// Each profile defines per-tier TQ bit width + weight set for systems
// where KV cache stays in unified memory (no GPU/CPU split).
// TQ bits: 0 = F16 (no compression), 2-6 = TQ encode at that bit width.

typedef struct {
    smart_kv_weights weights;
    int tq_bits[SMART_KV_TIER_COUNT];  // TQ bits per tier (0=F16)
    const char * name;
} smart_kv_tq_profile;

// High quality: near-lossless, ~2× compression
//   Tier 1-2: F16, Tier 3-4: TQ-5 (0.994), Tier 5: TQ-4 (0.969), Tier 6: TQ-2 (0.941)
extern const smart_kv_tq_profile SMART_KV_TQ_UNITY_HIGH;

// Balanced: good savings, ~3× compression
//   Tier 1: F16, Tier 2-3: TQ-5, Tier 4: TQ-4, Tier 5-6: TQ-2
extern const smart_kv_tq_profile SMART_KV_TQ_UNITY_BALANCED;

// Performance: aggressive compression, ~4×
//   Tier 1-2: TQ-5, Tier 3-4: TQ-4, Tier 5-6: TQ-2
extern const smart_kv_tq_profile SMART_KV_TQ_UNITY_PERF;

// Ultra: max compression, ~5×
//   All tiers: TQ-2
extern const smart_kv_tq_profile SMART_KV_TQ_UNITY_ULTRA;

// Apply a TQ profile (copies weights + tq_bits into the config)
void smart_kv_apply_tq_profile(smart_kv_config * cfg, const smart_kv_tq_profile * profile);

// Default weight sets
extern const smart_kv_weights SMART_KV_WEIGHTS_NO_ATTN;
extern const smart_kv_weights SMART_KV_WEIGHTS_ATTN;

// Init: precompute thresholds from gamma, fill default tag mods
void smart_kv_init_weights(smart_kv_weights * w, float gamma);

// Init tag mods to 1.0 (no modification)
void smart_kv_init_tag_mods(smart_kv_weights * w);

// Adaptive gamma based on memory pressure
static inline float smart_kv_adaptive_gamma(float base_gamma, uint32_t used, uint32_t capacity) {
    if (capacity == 0) return base_gamma;
    float pressure = (float)used / (float)capacity;
    if (pressure > 0.85f) return base_gamma * 0.7f;
    if (pressure > 0.70f) return base_gamma * 0.85f;
    return base_gamma;
}

// Returns true if memory pressure exceeds the offload threshold
static inline bool smart_kv_should_offload(uint32_t used, uint32_t capacity, float score) {
    if (capacity == 0) return false;
    float pressure = (float)used / (float)capacity;
    float threshold = 0.05f;
    if (pressure > 0.92f) threshold = 0.20f;
    else if (pressure > 0.85f) threshold = 0.12f;
    else if (pressure > 0.75f) threshold = 0.10f;
    else if (pressure > 0.65f) threshold = 0.08f;
    else if (pressure > 0.50f) threshold = 0.05f;
    return score <= threshold;
}

// --- Inline hot-path helpers ---

static inline float smart_kv_recency(uint64_t now, uint64_t last, float tau_r) {
    if (tau_r <= 0.0f) return 0.0f;
    if (now <= last) return 1.0f;
    int64_t d = (int64_t)(now - last);
    if (d >= (int64_t)(tau_r * 6.0f)) return 0.0f;
    return expf(-(float)d / tau_r);
}

static inline float smart_kv_frequency(uint32_t cnt, uint32_t max_cnt) {
    if (max_cnt == 0 || cnt == 0) return 0.0f;
    return log1pf((float)cnt) / log1pf((float)max_cnt);
}

static inline float smart_kv_pin_score(bool pinned, float anchor) {
    return pinned ? 1.0f : (anchor > 0.0f ? anchor : 0.0f);
}

static inline float smart_kv_tag_mod(float base_score, float tag_mod) {
    return base_score * tag_mod;
}

// Score one chunk with tag-aware modifiers
static inline float smart_kv_score(
    const smart_kv_chunk_meta * c,
    uint64_t now,
    uint32_t max_access,
    const smart_kv_weights * w)
{
    uint8_t t = c->tag < SMART_KV_MAX_TAGS ? c->tag : 0;

    float R = smart_kv_tag_mod(smart_kv_recency(now, c->last_used_step, w->tau_r), w->tag_mod_recency[t]);
    float A = c->attention_ema;
    float F = smart_kv_tag_mod(smart_kv_frequency(c->access_count, max_access), w->tag_mod_frequency[t]);
    float Q = c->query_score > 0.0f ? c->query_score : 0.5f;
    float P = smart_kv_tag_mod(smart_kv_pin_score(c->pinned, c->anchor_score), w->tag_mod_pin[t]);
    float D = smart_kv_tag_mod(c->redundancy_score, w->tag_mod_redundancy[t]);

    float s = w->w_recency * R + w->w_attention * A + w->w_frequency * F
            + w->w_query * Q + w->w_pin * P - w->w_redundancy * D;
    if (s < 0.0f) s = 0.0f;
    if (s > 1.0f) s = 1.0f;
    return s;
}

// Assign tier respecting min_tier
static inline smart_kv_tier smart_kv_assign_tier(float priority, uint8_t min_tier,
    const float thresholds[SMART_KV_TIER_COUNT])
{
    smart_kv_tier t;
    if (priority >= thresholds[0]) t = SMART_KV_TIER_VERY_HIGH;
    else if (priority >= thresholds[1]) t = SMART_KV_TIER_QUALITY;
    else if (priority >= thresholds[2]) t = SMART_KV_TIER_BALANCED;
    else if (priority >= thresholds[3]) t = SMART_KV_TIER_PERFORMANCE;
    else if (priority >= thresholds[4]) t = SMART_KV_TIER_ULTRA;
    else t = SMART_KV_TIER_ULTRA_TQ;

    // Enforce min_tier (1-based, 0 = no restriction)
    if (min_tier > 0 && (uint8_t)t > min_tier) t = (smart_kv_tier)min_tier;
    return t;
}

// Returns true if this tier uses the TurboQuant plugin (CPU offload)
static inline bool smart_kv_tier_is_plugin(smart_kv_tier t) {
    return t == SMART_KV_TIER_ULTRA_TQ;
}

// Migration backoff: returns true if this chunk should be skipped
static inline bool smart_kv_should_skip(uint32_t skip_counter, uint32_t min_residency, uint64_t now, uint64_t last_migrated) {
    if (skip_counter > 0) { return true; }  // under backoff
    if ((now - last_migrated) < min_residency) { return true; }
    return false;
}

// --- Prefill Analysis ---

// Hint derived from text content before the chunk is stored.
// Uses only static signals detectable during prefill (no recency/attention).
typedef struct {
    uint8_t  tag;               // smart_kv_tag classification
    uint8_t  min_tier;          // floor enforced from content type
    bool     pinned;
    float    anchor_score;      // heuristic pin boost (0-1)
    float    redundancy_score;  // boilerplate/low-info penalty (0-1)
    float    confidence;        // 0-1: how sure the analysis is
} smart_kv_prefill_hint;

// Lightweight text analysis: scans up to `len` bytes for known patterns.
// Returns a hint with tag, min_tier, pinned, redundancy, and confidence.
// Designed to run on chunk text during prefill before KV is stored.
smart_kv_prefill_hint smart_kv_prefill_analyze(const char * text, uint32_t len);

// Score a prefill hint using only static weights (pin, redundancy, anchor).
// No recency/attention/frequency — those need runtime data.
// Result is a priority 0-1. Calls smart_kv_assign_tier internally.
static inline float smart_kv_prefill_score(const smart_kv_prefill_hint * h,
    const smart_kv_weights * w)
{
    uint8_t t = h->tag < SMART_KV_MAX_TAGS ? h->tag : 0;
    float P = smart_kv_tag_mod(smart_kv_pin_score(h->pinned, h->anchor_score), w->tag_mod_pin[t]);
    float D = smart_kv_tag_mod(h->redundancy_score, w->tag_mod_redundancy[t]);
    float s = w->w_pin * P - w->w_redundancy * D;
    if (s < 0.0f) s = 0.0f;
    if (s > 1.0f) s = 1.0f;
    return s;
}

// Assign initial tier from a prefill hint. Uses pin + redundancy only.
static inline smart_kv_tier smart_kv_prefill_tier(const smart_kv_prefill_hint * h,
    const smart_kv_config * cfg)
{
    float s = smart_kv_prefill_score(h, &cfg->weights);
    return smart_kv_assign_tier(s, h->min_tier, cfg->weights.priority_thresholds);
}

// --- API ---

const char * smart_kv_tier_name(smart_kv_tier t);
const char * smart_kv_tag_name(uint8_t tag);

int smart_kv_tier_to_type(smart_kv_tier tier, int base_type);
smart_kv_scored_chunk smart_kv_eval(const smart_kv_chunk_meta * c, uint64_t now,
    uint32_t max_access, const smart_kv_config * cfg);
void smart_kv_log(const smart_kv_chunk_meta * c, float score, smart_kv_tier t, int target);

#ifdef __cplusplus
}
#endif
