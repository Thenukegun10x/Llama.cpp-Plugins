// Smart-TQ Cache Plugin: smart scoring + TurboQuant CPU offload for tier 6
// Provides: GGML_PLUGIN_CAP_KV_CACHE (v1)
//
// Tiers 1-5: stored as F16 on GPU (standard path)
// Tier 6:    stored as TurboQuant on CPU only — VRAM-free
//
// The F16 GPU tensor is still allocated but tier-6 slots leave it ZEROED
// so it consumes only the fixed allocation, not per-slot memory pressure.
// True VRAM savings require the mixed-cache routing patch (see below).

#include "ggml-plugin.h"
#include "ggml.h"
#include "../turbo-quant-plugin/turboquant.h"  // PolarQuant+QJL
#include "smart-kv-cache.h"                     // scoring + tier assignment
#include "learned-score.h"                      // MLP scorer

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <vector>
#include <algorithm>

// ── FP16 helpers ──────────────────────────────────────────────────────

static inline uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t s = (bits >> 16) & 0x8000;
    int32_t e = ((bits >> 23) & 0xFF) - 112;
    if (e > 31) e = 31;
    if (e < 0) e = 0;
    uint32_t m = (bits >> 13) & 0x3FF;
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (uint32_t)(h & 0x7C00u) >> 10;
    uint32_t m = (uint32_t)(h & 0x03FFu);
    if (e == 0) {
        if (m == 0) return 0.0f;
        int shift = 0;
        while ((m & 0x0400u) == 0) { m <<= 1; shift++; }
        m &= 0x03FFu;
        e = 113 - (uint32_t)shift;
    } else {
        e += 112;
    }
    uint32_t bits = s | (e << 23) | (m << 13);
    float result;
    memcpy(&result, &bits, sizeof(bits));
    return result;
}

// ── Per-layer context ────────────────────────────────────────────────

struct SmartTQContext {
    // Learned scorers (QuantRegressor + RAMDemoter)
    learned_scorer_t    lscorer;
    ls_ring_buffer_t    train_ring;

    // Model geometry
    int64_t n_embd_k_gqa;
    int64_t n_embd_v_gqa;
    int64_t kv_size;
    int32_t n_head_kv;
    int32_t head_dim_k;
    int32_t head_dim_v;
    int     bits_per_angle;

    // TQ storage per-slot: k_data[v][slot] and v_data[v][slot] for head v at slot pos
    // Indirection: for each slot, store whether it's tier 6 (TQ only) or tier 1-5 (F16)
    uint8_t*  slot_tier;       // [kv_size] — 0 = F16 (tiers 1-5), 6 = Ultra-TQ
    uint8_t*  k_data;          // [kv_size * n_head_kv * tq_bytes]
    uint8_t*  v_data;          // [kv_size * n_head_kv * tq_bytes]
    size_t    tq_bytes;        // bytes per head vector in TQ format

    // Smart cache scoring state
    smart_kv_config   cfg;
    smart_kv_chunk_meta* chunks;  // [kv_size] — one per slot
    uint64_t          step;

    // Stats
    uint64_t n_tier6_stored;
    uint64_t n_tier6_retrieved;

    SmartTQContext() : n_tier6_stored(0), n_tier6_retrieved(0) {}
};

// ── Lifecycle ────────────────────────────────────────────────────────

static bool g_loaded = false;

static void on_load(void) {
    g_loaded = true;
    fprintf(stderr, "[smart-tq] Smart + TurboQuant cache plugin loaded\n");
}

static void on_unload(void) {
    g_loaded = false;
    fprintf(stderr, "[smart-tq] unloaded\n");
}

static void on_model_loaded(const ggml_plugin_model_info_t* info,
                            size_t* scratch_size_out) {
    fprintf(stderr, "[smart-tq] model: %ld layers, %ld heads, hd=%ld\n",
            (long)info->n_layers, (long)info->n_head_kv, (long)info->head_dim);
    *scratch_size_out = 0;
}

// ── kv_cache_supported ──────────────────────────────────────────────

static bool kv_cache_supported(const ggml_plugin_kv_cache_params_t* params,
                                int bits_per_angle) {
    if (bits_per_angle < 1 || bits_per_angle > 16) return false;
    if (!turboquant_valid_dims(params->head_dim_k)) return false;
    if (!turboquant_valid_dims(params->head_dim_v)) return false;
    return true;
}

// ── Layer create / destroy ──────────────────────────────────────────

static void* kv_cache_create_layer(const ggml_plugin_kv_cache_params_t* params,
                                    int bits_per_angle) {
    SmartTQContext* ctx = new (std::nothrow) SmartTQContext();
    if (!ctx) return NULL;

    ctx->n_embd_k_gqa  = params->n_embd_k_gqa;
    ctx->n_embd_v_gqa  = params->n_embd_v_gqa;
    ctx->kv_size       = params->kv_size;
    ctx->n_head_kv     = params->n_head_kv;
    ctx->head_dim_k    = params->head_dim_k;
    ctx->head_dim_v    = params->head_dim_v;
    ctx->bits_per_angle = bits_per_angle;

    // TQ block size = larger of K/V head dim
    int hd = (params->head_dim_k > params->head_dim_v)
             ? params->head_dim_k : params->head_dim_v;
    ctx->tq_bytes = turboquant_size(hd, bits_per_angle);

    // Allocate TQ CPU storage
    size_t total = (size_t)ctx->kv_size * (size_t)ctx->n_head_kv * ctx->tq_bytes;
    ctx->k_data = (uint8_t*)calloc(total, 1);
    ctx->v_data = (uint8_t*)calloc(total, 1);
    ctx->slot_tier = (uint8_t*)calloc((size_t)ctx->kv_size, 1);

    // Allocate chunk metadata for smart scoring
    ctx->chunks = (smart_kv_chunk_meta*)calloc((size_t)ctx->kv_size,
                    sizeof(smart_kv_chunk_meta));

    if (!ctx->k_data || !ctx->v_data || !ctx->slot_tier || !ctx->chunks) {
        free(ctx->k_data); free(ctx->v_data);
        free(ctx->slot_tier); free(ctx->chunks);
        delete ctx;
        return NULL;
    }

    // Initialize smart cache config
    ctx->cfg.weights = SMART_KV_WEIGHTS_NO_ATTN;
    smart_kv_init_weights(&ctx->cfg.weights, ctx->cfg.weights.gamma);
    smart_kv_init_tag_mods(&ctx->cfg.weights);
    ctx->cfg.score_interval = 512;
    ctx->cfg.migrate_max = 8;
    ctx->cfg.base_type = GGML_TYPE_Q4_K;
    ctx->cfg.memory_capacity = (uint32_t)ctx->kv_size;
    ctx->cfg.memory_used = 0;
    ctx->step = 0;

    // Initialize chunk metadata
    for (int64_t i = 0; i < ctx->kv_size; i++) {
        ctx->chunks[i].chunk_id = (uint32_t)i;
        ctx->chunks[i].tag = SMART_KV_TAG_DEFAULT;
        ctx->chunks[i].pinned = false;
        ctx->chunks[i].attention_ema = 0.5f;
        ctx->chunks[i].query_score = 0.5f;
        ctx->chunks[i].anchor_score = 0.0f;
        ctx->chunks[i].redundancy_score = 0.0f;
        ctx->chunks[i].skip_counter = 0;
        ctx->chunks[i].min_tier = 0;
    }

    // Initialize learned scorers + ring buffer
    memset(&ctx->lscorer, 0, sizeof(ctx->lscorer));
    ls_init_random(&ctx->lscorer, 42);
    ls_ring_init(&ctx->train_ring);

    // Try to load pre-trained weights (optional)
    {
        // First try plugin directory
        ls_load_all_weights(&ctx->lscorer, ".");
        if (!ctx->lscorer.quant_loaded) {
            // Try plugin-relative path
            ls_load_all_weights(&ctx->lscorer, "plugins/smart-kv-plugin");
        }
        if (ctx->lscorer.quant_loaded) {
            fprintf(stderr, "[smart-tq] QuantRegressor ready\n");
        }
        if (ctx->lscorer.ram_loaded) {
            fprintf(stderr, "[smart-tq] RAMDemoter ready\n");
        }
    }

    fprintf(stderr, "[smart-tq] layer %d: %ld slots x %d heads, "
            "TQ block=%zuB, scoring table=%zuB\n",
            params->layer_idx, (long)ctx->kv_size, ctx->n_head_kv,
            ctx->tq_bytes, total);

    return ctx;
}

static void kv_cache_destroy_layer(void* layer_ctx) {
    if (!layer_ctx) return;
    SmartTQContext* ctx = (SmartTQContext*)layer_ctx;
    free(ctx->k_data);
    free(ctx->v_data);
    free(ctx->slot_tier);
    free(ctx->chunks);
    delete ctx;
}

// ── K/V norm helpers ─────────────────────────────────────────────────

static float compute_k_norm_head(const uint16_t* fp16, int head_dim) {
    double sum_sq = 0.0;
    for (int i = 0; i < head_dim; i++)
        sum_sq += (double)f16_to_f32(fp16[i]) * (double)f16_to_f32(fp16[i]);
    return (float)(sqrt(sum_sq) / sqrt((double)head_dim));
}

// ── Encode one head vector to TQ ─────────────────────────────────────

static void encode_head_fp16(const uint16_t* fp16_src, int head_dim,
                              uint8_t* dst, int bits) {
    float* tmp = (float*)malloc((size_t)head_dim * sizeof(float));
    for (int i = 0; i < head_dim; i++)
        tmp[i] = f16_to_f32(fp16_src[i]);
    turboquant_encode(tmp, head_dim, dst, bits, 0);
    free(tmp);
}

static void decode_head_fp16(const uint8_t* src, int head_dim,
                              uint16_t* fp16_dst, int bits) {
    float* tmp = (float*)malloc((size_t)head_dim * sizeof(float));
    turboquant_decode(src, head_dim, tmp, bits, 0);
    for (int i = 0; i < head_dim; i++)
        fp16_dst[i] = f32_to_f16(tmp[i]);
    free(tmp);
}

// ── Score a slot — returns tier ─────────────────────────────────────
// Uses QuantRegressor for tier 1-5.
// Tier 6 (RAM eviction) is always decided by heuristic scorer during data
// collection — RAMDemoter trains on these labels to learn counterfactually.
// Once trained, RAMDemoter inference can be enabled (see ARCHITECTURE.md).

static uint8_t score_slot(SmartTQContext* ctx, int64_t slot) {
    if (slot < 0 || slot >= ctx->kv_size) return 0;
    smart_kv_chunk_meta* c = &ctx->chunks[slot];
    uint32_t max_access = 1;
    for (int64_t i = 0; i < ctx->kv_size; i++)
        if (ctx->chunks[i].access_count > max_access)
            max_access = ctx->chunks[i].access_count;

    ctx->cfg.memory_used = 0;
    for (int64_t i = 0; i < ctx->kv_size; i++)
        if (ctx->slot_tier[i] != 0 || ctx->chunks[i].access_count > 0)
            ctx->cfg.memory_used++;

    // Quant scoring via QuantRegressor or heuristic fallback
    if (ctx->lscorer.quant_loaded) {
        ls_features_t feat = ls_features_from_meta(c, ctx->step, (uint32_t)ctx->kv_size, ctx->cfg.weights.gamma);
        float q_score = qr_forward(&ctx->lscorer.quant, &feat);
        float gamma = smart_kv_adaptive_gamma(ctx->cfg.weights.gamma, ctx->cfg.memory_used, ctx->cfg.memory_capacity);
        smart_kv_weights adj = ctx->cfg.weights;
        if (gamma != adj.gamma) {
            smart_kv_init_weights(&adj, gamma);
        }
        smart_kv_tier t = smart_kv_assign_tier(q_score, c->min_tier, adj.priority_thresholds);
        return (uint8_t)t;
    }

    // Heuristic scoring (fallback)
    smart_kv_scored_chunk r = smart_kv_eval(c, ctx->step, max_access, &ctx->cfg);
    return (uint8_t)r.tier;
}

// ── kv_cache_store ──────────────────────────────────────────────────

static int kv_cache_store(void* layer_ctx, int64_t pos,
                           const void* k_data, const void* v_data,
                           int64_t n_tokens) {
    SmartTQContext* ctx = (SmartTQContext*)layer_ctx;
    if (!ctx || !k_data || !v_data) return -1;

    const uint16_t* k_fp16 = (const uint16_t*)k_data;
    const uint16_t* v_fp16 = (const uint16_t*)v_data;
    size_t row_bytes_k = (size_t)ctx->n_head_kv * ctx->tq_bytes;
    size_t row_bytes_v = (size_t)ctx->n_head_kv * ctx->tq_bytes;

    for (int64_t t = 0; t < n_tokens; t++) {
        int64_t slot = pos + t;
        if (slot >= ctx->kv_size) return -1;

        ctx->step++;

        // Update chunk metadata
        smart_kv_chunk_meta* c = &ctx->chunks[slot];
        c->last_used_step = ctx->step;
        c->access_count++;

        // Compute K/V norms — cheap, most predictive features
        double k_sum_sq = 0.0, v_sum_sq = 0.0;
        for (int32_t h = 0; h < ctx->n_head_kv; h++) {
            int off_k = (int)((size_t)t * ctx->n_embd_k_gqa + (size_t)h * ctx->head_dim_k);
            int off_v = (int)((size_t)t * ctx->n_embd_v_gqa + (size_t)h * ctx->head_dim_v);
            k_sum_sq += compute_k_norm_head(k_fp16 + off_k, ctx->head_dim_k);
            v_sum_sq += compute_k_norm_head(v_fp16 + off_v, ctx->head_dim_v);
        }
        float k_norm_tok = (float)(k_sum_sq / ctx->n_head_kv);
        float v_norm_tok = (float)(v_sum_sq / ctx->n_head_kv);

        // Welford online update for running mean + variance
        if (c->n_tokens == 0) {
            c->k_norm = k_norm_tok;
            c->v_norm = v_norm_tok;
            c->k_variance = 0.0f;
        } else {
            float prev_mean = c->k_norm;
            float delta = k_norm_tok - prev_mean;
            c->k_norm += delta / (float)(c->n_tokens + 1);
            c->k_variance += delta * (k_norm_tok - c->k_norm);
            // V norm: running mean only (variance less useful)
            c->v_norm += (v_norm_tok - c->v_norm) / (float)(c->n_tokens + 1);
        }
        c->n_tokens++;

        float attn = 0.5f + 0.5f * ((float)(rand() % 100) / 100.0f);
        float lambda_a = ctx->cfg.weights.lambda_a;
        c->attention_ema = lambda_a * c->attention_ema + (1.0f - lambda_a) * attn;

        // Score to determine tier (uses QuantRegressor or heuristic)
        uint8_t tier = score_slot(ctx, slot);

        // Collect training sample (features + labels)
        // RAM label uses heuristic tier to break circularity:
        //   heuristic_tier = what the heuristic scorer would assign
        //   ram_label = 1.0 if heuristic says tier 6 (will be corrected by fixup_labels if re-accessed)
        {
            ls_features_t feat = ls_features_from_meta(c, ctx->step, (uint32_t)ctx->kv_size, ctx->cfg.weights.gamma);
            float mem_pressure = ctx->cfg.memory_capacity > 0
                ? (float)ctx->cfg.memory_used / (float)ctx->cfg.memory_capacity
                : 0.0f;

            // Quant label: normalized tier (0.2 = tier 5, 1.0 = tier 1)
            float quant_label = 1.0f - ((float)(tier - 1) / (float)(SMART_KV_TIER_COUNT - 1));
            if (tier >= SMART_KV_TIER_ULTRA_TQ) quant_label = 0.0f;

            // Heuristic tier for RAM label (independent of QuantRegressor)
            float heuristic_tier_f = 0;
            {
                uint32_t max_a = 1;
                for (int64_t i = 0; i < ctx->kv_size; i++)
                    if (ctx->chunks[i].access_count > max_a) max_a = ctx->chunks[i].access_count;
                smart_kv_scored_chunk hr = smart_kv_eval(c, ctx->step, max_a, &ctx->cfg);
                heuristic_tier_f = (float)hr.tier;
            }
            uint8_t ht = (uint8_t)heuristic_tier_f;
            float ram_label = (ht >= SMART_KV_TIER_ULTRA_TQ && !c->pinned) ? 1.0f : 0.0f;

            ls_ring_push(&ctx->train_ring, &feat, quant_label, ram_label, mem_pressure,
                         c->chunk_id, c->access_count, ht);

            // Auto-export at 1000, 4000 samples
            if (ctx->train_ring.count == 1000 || ctx->train_ring.count == 4000) {
                ls_ring_fixup_labels(&ctx->train_ring, ctx->chunks, (uint32_t)ctx->kv_size);
                char dpath[256];
                snprintf(dpath, sizeof(dpath), "training_data_%u.bin", ctx->train_ring.count);
                int n = ls_export_dataset(dpath, &ctx->train_ring);
                fprintf(stderr, "[smart-tq] exported %d samples to %s\n", n, dpath);
            }
        }

        if (tier == SMART_KV_TIER_ULTRA_TQ) {
            // ── Tier 6: TQ-encode on CPU only ──
            ctx->slot_tier[slot] = SMART_KV_TIER_ULTRA_TQ;
            uint8_t* k_row = ctx->k_data + slot * row_bytes_k;
            uint8_t* v_row = ctx->v_data + slot * row_bytes_v;

            for (int32_t h = 0; h < ctx->n_head_kv; h++) {
                int off_k = (int)((size_t)t * ctx->n_embd_k_gqa + (size_t)h * ctx->head_dim_k);
                int off_v = (int)((size_t)t * ctx->n_embd_v_gqa + (size_t)h * ctx->head_dim_v);
                encode_head_fp16(k_fp16 + off_k, ctx->head_dim_k,
                                 k_row + (size_t)h * ctx->tq_bytes, ctx->bits_per_angle);
                encode_head_fp16(v_fp16 + off_v, ctx->head_dim_v,
                                 v_row + (size_t)h * ctx->tq_bytes, ctx->bits_per_angle);
            }
            ctx->n_tier6_stored++;

        } else {
            // ── Tier 1-5: store as F16 (GPU tensor handles this) ──
            // The GPU F16 tensor was already written by llama.cpp.
            // We just mark the slot as F16-tier.
            ctx->slot_tier[slot] = 0;  // 0 = GPU F16
        }

        c->tier = tier;
        c->target_tier = tier;
    }

    return 0;
}

// ── kv_cache_retrieve ───────────────────────────────────────────────
// Also bumps access_count — a retrieved chunk is still "used" and should
// not be evicted. Tier-6 chunks that are retrieved are marked for promotion
// (slot_tier cleared so next scoring cycle assigns a proper GPU tier).

static int kv_cache_retrieve(void* layer_ctx, int64_t pos,
                              void* k_out, void* v_out,
                              int64_t n_tokens) {
    SmartTQContext* ctx = (SmartTQContext*)layer_ctx;
    if (!ctx || !k_out || !v_out) return -1;

    uint16_t* k_fp16 = (uint16_t*)k_out;
    uint16_t* v_fp16 = (uint16_t*)v_out;
    size_t row_bytes_k = (size_t)ctx->n_head_kv * ctx->tq_bytes;
    size_t row_bytes_v = (size_t)ctx->n_head_kv * ctx->tq_bytes;

    for (int64_t t = 0; t < n_tokens; t++) {
        int64_t slot = pos + t;
        if (slot >= ctx->kv_size) {
            memset(k_fp16 + (size_t)t * ctx->n_embd_k_gqa, 0,
                   (size_t)ctx->n_embd_k_gqa * sizeof(uint16_t));
            memset(v_fp16 + (size_t)t * ctx->n_embd_v_gqa, 0,
                   (size_t)ctx->n_embd_v_gqa * sizeof(uint16_t));
            continue;
        }

        // Bump access tracking on every retrieve (not just store)
        smart_kv_chunk_meta* c = &ctx->chunks[slot];
        ctx->step++;
        c->last_used_step = ctx->step;
        c->access_count++;

        if (ctx->slot_tier[slot] == SMART_KV_TIER_ULTRA_TQ) {
            // ── Tier 6: decode from TQ CPU, mark for promotion ──
            // Clear slot_tier so the next scoring cycle promotes this chunk
            // back to an appropriate GPU tier.
            ctx->slot_tier[slot] = 0;
            c->tier = 0;
            c->target_tier = 0;

            uint8_t* k_row = ctx->k_data + slot * row_bytes_k;
            uint8_t* v_row = ctx->v_data + slot * row_bytes_v;

            for (int32_t h = 0; h < ctx->n_head_kv; h++) {
                int off_k = (int)((size_t)t * ctx->n_embd_k_gqa + (size_t)h * ctx->head_dim_k);
                int off_v = (int)((size_t)t * ctx->n_embd_v_gqa + (size_t)h * ctx->head_dim_v);
                decode_head_fp16(k_row + (size_t)h * ctx->tq_bytes, ctx->head_dim_k,
                                 k_fp16 + off_k, ctx->bits_per_angle);
                decode_head_fp16(v_row + (size_t)h * ctx->tq_bytes, ctx->head_dim_v,
                                 v_fp16 + off_v, ctx->bits_per_angle);
            }
            ctx->n_tier6_retrieved++;

        } else {
            // ── Tier 1-5: data is already in the F16 GPU tensor ──
            memcpy(k_fp16 + (size_t)t * ctx->n_embd_k_gqa,
                   ctx->chunks[slot].chunk_id ? k_fp16 : k_fp16,
                   (size_t)ctx->n_embd_k_gqa * sizeof(uint16_t));
            memcpy(v_fp16 + (size_t)t * ctx->n_embd_v_gqa,
                   ctx->chunks[slot].chunk_id ? v_fp16 : v_fp16,
                   (size_t)ctx->n_embd_v_gqa * sizeof(uint16_t));
        }
    }

    return 0;
}

// ── kv_cache_clear ──────────────────────────────────────────────────

static void kv_cache_clear(void* layer_ctx) {
    SmartTQContext* ctx = (SmartTQContext*)layer_ctx;
    if (!ctx) return;

    // Export training data before clearing (with counterfactual RAM labels)
    if (ctx->train_ring.count > 100) {
        ls_ring_fixup_labels(&ctx->train_ring, ctx->chunks, (uint32_t)ctx->kv_size);
        char dpath[256];
        time_t now = time(NULL);
        struct tm * tm_info = localtime(&now);
        strftime(dpath, sizeof(dpath), "session_%Y%m%d_%H%M%S.bin", tm_info);
        int n = ls_export_dataset(dpath, &ctx->train_ring);
        fprintf(stderr, "[smart-tq] session ended, exported %d samples to %s\n", n, dpath);
    }

    size_t total = (size_t)ctx->kv_size * (size_t)ctx->n_head_kv * ctx->tq_bytes;
    memset(ctx->k_data, 0, total);
    memset(ctx->v_data, 0, total);
    memset(ctx->slot_tier, 0, (size_t)ctx->kv_size);
    memset(ctx->chunks, 0, (size_t)ctx->kv_size * sizeof(smart_kv_chunk_meta));

    // Re-init chunk metadata
    for (int64_t i = 0; i < ctx->kv_size; i++) {
        ctx->chunks[i].chunk_id = (uint32_t)i;
        ctx->chunks[i].attention_ema = 0.5f;
        ctx->chunks[i].query_score = 0.5f;
    }

    ctx->step = 0;
    ctx->n_tier6_stored = 0;
    ctx->n_tier6_retrieved = 0;

    // Re-init ring buffer for next session
    ls_ring_init(&ctx->train_ring);

    fprintf(stderr, "[smart-tq] cache cleared\n");
}

// ═══════════════════════════════════════════════════════════════════
// CAPABILITY DECLARATION
// ═══════════════════════════════════════════════════════════════════

static ggml_plugin_kv_cache_v1_t g_kv_cache_v1 = {
    .version             = 1,
    .on_load             = on_load,
    .on_model_loaded     = on_model_loaded,
    .on_unload           = on_unload,
    .kv_cache_supported   = kv_cache_supported,
    .kv_cache_create_layer = kv_cache_create_layer,
    .kv_cache_destroy_layer = kv_cache_destroy_layer,
    .kv_cache_store      = kv_cache_store,
    .kv_cache_retrieve   = kv_cache_retrieve,
    .kv_cache_clear      = kv_cache_clear,
};

static const ggml_plugin_capability_t g_caps[] = {
    { GGML_PLUGIN_CAP_KV_CACHE, 1, &g_kv_cache_v1 },
};

static ggml_plugin_descriptor_t g_desc = {
    .api_version      = GGML_PLUGIN_API_VERSION,
    .name             = "smart-tq-cache",
    .version          = "0.1.0",
    .description      = "Smart tiered KV cache: tiers 1-5 GPU F16, tier 6 TurboQuant CPU offload",
    .num_capabilities = 1,
    .capabilities     = g_caps,
};

extern "C" GGML_PLUGIN_EXPORT
ggml_plugin_descriptor_t* ggml_plugin_get_descriptor(void) {
    return &g_desc;
}
