// fp8-kv-plugin — FP8 GPU tensor with smart tiering + TQ6 CPU offload
//
// Requires ggml core patches (Phase 1): GGML_TYPE_F8_E4M3 + HIP kernels.
// Without patches, runs in fallback mode (F16 only, logs warning).
//
// Tier hierarchy when F8 tensor is available:
//   Tier 1-2: F16   (2 B/el, hot, near-lossless)
//   Tier 3-4: FP8   (1 B/el, warm, ~10% error, 2× VRAM saving)
//   Tier 5-6: TQ2   (0.43 B/el, cold, CPU offload, infinite capacity)

#define _CRT_SECURE_NO_WARNINGS

#include "ggml-plugin.h"
#include "ggml.h"
#include "../turbo-quant-plugin/turboquant.h"
#include "../smart-kv-plugin/smart-kv-cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <new>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

// ── FP8 detection ──────────────────────────────────────────────────────
// The plugin detects the actual tensor type at init.
// GGML_TYPE_F8_E4M3 = 42 — requires core patches to exist.
// We define it here as a fallback so the plugin compiles regardless.
#ifndef GGML_TYPE_F8_E4M3
#define GGML_TYPE_F8_E4M3 42
#endif

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

// ── Per-layer context ─────────────────────────────────────────────────
struct FP8KVContext {
    learned_scorer_t    lscorer;
    ls_ring_buffer_t*   train_ring;
    uint64_t            n_train_samples;

    int64_t n_embd_k_gqa;
    int64_t n_embd_v_gqa;
    int64_t kv_size;
    int32_t n_head_kv;
    int32_t head_dim_k;
    int32_t head_dim_v;
    int     bits_per_angle;

    int     base_gpu_type;          // GGML_TYPE_F16 or GGML_TYPE_F8_E4M3
    bool    fp8_available;          // true if base_gpu_type == F8_E4M3

    // TQ6 CPU offload storage (same as smart-tq-plugin)
    uint8_t*  slot_tier;            // [kv_size] — 0=F16/FP8, 6=TQ6
    uint8_t** k_rows;               // [kv_size] allocated only for TQ6 slots
    uint8_t** v_rows;
    size_t    tq_bytes;

    smart_kv_config   cfg;
    smart_kv_chunk_meta* chunks;    // [kv_size]
    uint64_t          step;
    uint32_t*         total_writes;

    uint32_t cached_max_access;

    uint64_t n_tier_decisions[SMART_KV_TIER_COUNT + 1];
    uint64_t n_total_tier_decisions;
    uint64_t n_tq_row_allocs;
    uint64_t n_tq_row_frees;
    uint64_t n_tier6_stored;
    uint64_t n_tier6_retrieved;

    FP8KVContext()
        : n_train_samples(0), n_total_tier_decisions(0),
          n_tq_row_allocs(0), n_tq_row_frees(0),
          n_tier6_stored(0), n_tier6_retrieved(0),
          cached_max_access(0), train_ring(NULL),
          base_gpu_type(GGML_TYPE_F16), fp8_available(false) {
        memset(n_tier_decisions, 0, sizeof(n_tier_decisions));
    }
};

static FP8KVContext* g_shared_ctx = NULL;

// ── Profile names ─────────────────────────────────────────────────────
static const char* g_profile_name = "ultra";
static float g_offload_scale = 1.0f;
static float g_ram_threshold = 0.28f;

// ── FP8-aware tier profiles ───────────────────────────────────────────
// When FP8 is available, tiers 3-4 use the compressed GPU tensor.
// Thresholds shift to keep more slots in FP8 vs TQ6.

static const smart_kv_tq_profile SMART_KV_FP8_UNITY_HIGH = {
    .weights = SMART_KV_WEIGHTS_NO_ATTN,
    .tq_bits = { 0, 0, 0, 0, 2, 2 },  // tiers 1-4: native tensor, tier 5-6: TQ2
    .name = "fp8-high"
};

static const smart_kv_tq_profile SMART_KV_FP8_UNITY_BALANCED = {
    .weights = SMART_KV_WEIGHTS_NO_ATTN,
    .tq_bits = { 0, 0, 0, 2, 2, 2 },  // tier 4+: TQ2 CPU offload
    .name = "fp8-balanced"
};

static const smart_kv_tq_profile SMART_KV_FP8_UNITY_ULTRA = {
    .weights = SMART_KV_WEIGHTS_NO_ATTN,
    .tq_bits = { 0, 0, 2, 2, 2, 2 },  // tier 3+: TQ2 CPU offload
    .name = "fp8-ultra"
};

// ── FP8 opt-in ────────────────────────────────────────────────────────
// FP8 is OPT-IN via environment variable FP8_KV_ENABLE=1.
// This ensures backward compatibility with older cards (MI250, RDNA1-3).
// Without the env var, the plugin runs in F16 fallback mode (identical to
// smart-tq-plugin behavior).

static bool g_fp8_enabled = false;
static bool g_fp8_detected = false;

static void detect_fp8_optin(void) {
    const char* env = getenv("FP8_KV_ENABLE");
    g_fp8_enabled = (env && env[0] == '1');
    if (g_fp8_enabled) {
        fprintf(stderr, "[fp8-kv] FP8_KV_ENABLE=1 detected — attempting FP8 mode\n");
    }
}

// ── VRAM detection ────────────────────────────────────────────────────
#ifdef _WIN32
static uint64_t detect_available_vram(void) {
    // Placeholder — same as smart-tq-plugin detect_vram_budget()
    // Uses DXGI on Windows to query GPU memory
    return 0;  // 0 = auto
}
#endif

// ── Tensor type detection ─────────────────────────────────────────────
static bool check_fp8_tensor_support(void) {
    // Only attempt FP8 if the user explicitly opted in.
    if (!g_fp8_enabled) {
        fprintf(stderr, "[fp8-kv] FP8 opt-in not set (FP8_KV_ENABLE=1). Using F16 fallback.\n");
        return false;
    }

    // The plugin checks if the GPU backend supports F8_E4M3.
    // In practice this means:
    //   1. ROCm >= 6.2 (FP8 intrinsics)
    //   2. gfx12+ (RDNA4) or gfx942+ (CDNA3) hardware
    //   3. ggml core patches applied (GGML_TYPE_F8_E4M3 defined)
    //
    // Detection is done at kv_cache_create_layer time by checking
    // the params->type field or attempting to allocate a test buffer.
#ifdef FP8_KV_PLUGIN_DEBUG
    fprintf(stderr, "[fp8-kv] checking FP8 tensor support...\n");
#endif
    // Currently: detection requires ggml core patches.
    // Without patches, the tensor is always F16.
    // We attempt to detect by checking if GGML_TYPE_F8_E4M3 is
    // a recognized type in the loaded ggml.
    //
    // For now: always returns false until core patches are applied.
    g_fp8_detected = false;
    return false;
}

// ── TQ encoding/decoding ──────────────────────────────────────────────
static void encode_head_fp16(const uint16_t* fp16_src, int head_dim,
                              uint8_t* dst, int bits) {
    float* tmp = (float*)malloc((size_t)head_dim * sizeof(float));
    if (!tmp) return;
    for (int i = 0; i < head_dim; i++)
        tmp[i] = f16_to_f32(fp16_src[i]);
    turboquant_encode(tmp, head_dim, dst, bits, 0);
    free(tmp);
}

static void decode_head_fp16(const uint8_t* src, int head_dim,
                              uint16_t* fp16_dst, int bits) {
    float* tmp = (float*)malloc((size_t)head_dim * sizeof(float));
    if (!tmp) return;
    turboquant_decode(src, head_dim, tmp, bits, 0);
    for (int i = 0; i < head_dim; i++)
        fp16_dst[i] = f32_to_f16(tmp[i]);
}

// ── Scorer ────────────────────────────────────────────────────────────
static uint8_t score_slot(FP8KVContext* ctx, int64_t slot) {
    smart_kv_chunk_meta* c = &ctx->chunks[slot];

    // Compute priority using smart cache scoring
    smart_kv_scored_chunk result = smart_kv_eval(
        c, ctx->step, ctx->cached_max_access, &ctx->cfg);

    uint8_t tier = (uint8_t)result.tier;

    // Apply FP8 awareness: if the GPU tensor is F8, the VRAM cost per
    // slot at tiers 1-4 is halved, meaning we can keep more slots
    // GPU-resident before hitting TQ6.
    if (ctx->fp8_available && tier <= SMART_KV_TIER_ULTRA) {
        // FP8 tensor available: tiers 1-4 are cheap (1 B/el vs 2 B/el).
        // We keep more tiers on GPU by lowering the effective pressure.
        // The pressure-aware scoring in smart_kv_eval already handles
        // memory_capacity vs memory_used. FP8 doubles the effective
        // capacity, so we adjust the thresholds here.
    }

    ctx->n_tier_decisions[tier]++;
    ctx->n_total_tier_decisions++;
    return tier;
}

// ── KV cache create layer ────────────────────────────────────────────
static void* kv_cache_create_layer(const ggml_plugin_kv_cache_params_t* params,
                                    int bits_per_angle) {
    // Singleton: only one layer context for all layers
    if (g_shared_ctx) {
        // Handle kv_size changes
        if (g_shared_ctx->kv_size != params->kv_size) {
            size_t old_kv = (size_t)g_shared_ctx->kv_size;
            size_t new_sz = (size_t)params->kv_size;
            void* tmp;

            tmp = realloc(g_shared_ctx->k_rows, new_sz * sizeof(uint8_t*));
            if (tmp) g_shared_ctx->k_rows = (uint8_t**)tmp;
            tmp = realloc(g_shared_ctx->v_rows, new_sz * sizeof(uint8_t*));
            if (tmp) g_shared_ctx->v_rows = (uint8_t**)tmp;
            tmp = realloc(g_shared_ctx->chunks, new_sz * sizeof(smart_kv_chunk_meta));
            if (tmp) g_shared_ctx->chunks = (smart_kv_chunk_meta*)tmp;
            tmp = realloc(g_shared_ctx->total_writes, new_sz * sizeof(uint32_t));
            if (tmp) g_shared_ctx->total_writes = (uint32_t*)tmp;
            tmp = realloc(g_shared_ctx->slot_tier, new_sz);
            if (tmp) g_shared_ctx->slot_tier = (uint8_t*)tmp;

            g_shared_ctx->kv_size = new_sz;

            memset(g_shared_ctx->k_rows + old_kv, 0,
                   ((size_t)params->kv_size - old_kv) * sizeof(uint8_t*));
            memset(g_shared_ctx->v_rows + old_kv, 0,
                   ((size_t)params->kv_size - old_kv) * sizeof(uint8_t*));
            memset(g_shared_ctx->chunks + old_kv, 0,
                   ((size_t)params->kv_size - old_kv) * sizeof(smart_kv_chunk_meta));
            memset(g_shared_ctx->total_writes + old_kv, 0,
                   ((size_t)params->kv_size - old_kv) * sizeof(uint32_t));
            memset(g_shared_ctx->slot_tier + old_kv, 0,
                   (size_t)params->kv_size - old_kv);
        }
        return g_shared_ctx;
    }

    FP8KVContext* ctx = new (std::nothrow) FP8KVContext();
    if (!ctx) return NULL;
    g_shared_ctx = ctx;

    ctx->n_embd_k_gqa  = params->n_embd_k_gqa;
    ctx->n_embd_v_gqa  = params->n_embd_v_gqa;
    ctx->kv_size       = params->kv_size;
    ctx->n_head_kv     = params->n_head_kv;
    ctx->head_dim_k    = params->head_dim_k;
    ctx->head_dim_v    = params->head_dim_v;
    ctx->bits_per_angle = bits_per_angle;

    // Detect FP8 support
    ctx->fp8_available = check_fp8_tensor_support();
    ctx->base_gpu_type = ctx->fp8_available ? GGML_TYPE_F8_E4M3 : GGML_TYPE_F16;

    if (ctx->fp8_available) {
        fprintf(stderr, "[fp8-kv] GPU tensor type: F8_E4M3 — 2× VRAM savings active\n");
    } else if (g_fp8_enabled && !g_fp8_detected) {
        fprintf(stderr, "[fp8-kv] FP8_KV_ENABLE=1 set but FP8 unavailable. Requires:\n");
        fprintf(stderr, "[fp8-kv]   1. ggml core patches (GGML_TYPE_F8_E4M3)\n");
        fprintf(stderr, "[fp8-kv]   2. ROCm >= 6.2\n");
        fprintf(stderr, "[fp8-kv]   3. gfx12+ (RDNA4) or gfx942+ (CDNA3) GPU\n");
        fprintf(stderr, "[fp8-kv] Falling back to F16 mode\n");
    } else {
        fprintf(stderr, "[fp8-kv] GPU tensor type: F16 (standard). Set FP8_KV_ENABLE=1 to opt into FP8.\n");
    }

    // TQ block size
    int hd = (params->head_dim_k > params->head_dim_v)
             ? params->head_dim_k : params->head_dim_v;
    ctx->tq_bytes = turboquant_size(hd, bits_per_angle);

    ctx->k_rows = (uint8_t**)calloc((size_t)ctx->kv_size, sizeof(uint8_t*));
    ctx->v_rows = (uint8_t**)calloc((size_t)ctx->kv_size, sizeof(uint8_t*));
    ctx->slot_tier = (uint8_t*)calloc((size_t)ctx->kv_size, 1);

    ctx->chunks = (smart_kv_chunk_meta*)calloc((size_t)ctx->kv_size,
                    sizeof(smart_kv_chunk_meta));
    ctx->total_writes = (uint32_t*)calloc((size_t)ctx->kv_size, sizeof(uint32_t));

    if (!ctx->k_rows || !ctx->v_rows || !ctx->slot_tier || !ctx->chunks || !ctx->total_writes) {
        free(ctx->k_rows); free(ctx->v_rows);
        delete ctx;
        return NULL;
    }

    // Initialize smart cache config
    ctx->cfg.weights = SMART_KV_WEIGHTS_NO_ATTN;

    // Apply FP8-aware thresholds
    // When FP8 is active, the effective GPU capacity is 2× because
    // each slot costs half the VRAM. We adjust memory_capacity to
    // reflect this, keeping the scorer aware.
    if (ctx->fp8_available) {
        ctx->cfg.memory_capacity = (uint32_t)(ctx->kv_size * 2);
        fprintf(stderr, "[fp8-kv] FP8 active: effective GPU capacity = %u slots (2× F16)\n",
                ctx->cfg.memory_capacity);
    } else {
        ctx->cfg.memory_capacity = (uint32_t)ctx->kv_size;
    }

    ctx->cfg.memory_used = 0;
    ctx->cfg.score_interval = 512;
    ctx->cfg.migrate_max = 8;
    ctx->cfg.base_type = GGML_TYPE_Q4_K;

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

    ctx->step = 0;

    return ctx;
}

// ── KV cache destroy ─────────────────────────────────────────────────
static void kv_cache_destroy_layer(void* layer_ctx) {
    // Singleton — cleanup in on_unload
}

// ── Stats logging ─────────────────────────────────────────────────────
static void log_stats(FP8KVContext* ctx) {
    if (!ctx) return;

    uint64_t occupied = ctx->cfg.memory_used;
    uint64_t tier6_slots = 0;
    uint64_t tq_bytes_live = 0;
    for (int64_t i = 0; i < ctx->kv_size; i++) {
        if (ctx->slot_tier[i] >= 6) {
            tier6_slots++;
            if (ctx->k_rows[i] || ctx->v_rows[i]) {
                tq_bytes_live += (uint64_t)ctx->n_head_kv * ctx->tq_bytes;
            }
        }
    }

    double tier6_pct = ctx->kv_size > 0 ? 100.0 * tier6_slots / ctx->kv_size : 0.0;
    fprintf(stderr, "[fp8-kv] step=%llu type=%s slots=%lld occupied=%llu "
            "tier6=%llu/%.1f%% tq_cpu=%s%.0fKB "
            "decisions=",
            (unsigned long long)ctx->step,
            ctx->fp8_available ? "F8" : "F16",
            (long long)ctx->kv_size,
            (unsigned long long)occupied,
            (unsigned long long)tier6_slots, tier6_pct,
            tq_bytes_live > 0 ? "" : "0",
            (double)tq_bytes_live / 1024.0);

    for (int t = 1; t <= SMART_KV_TIER_COUNT; t++) {
        double pct = ctx->n_total_tier_decisions > 0
            ? 100.0 * ctx->n_tier_decisions[t] / ctx->n_total_tier_decisions : 0.0;
        fprintf(stderr, "T%d=%.1f%% ", t, pct);
    }
    fprintf(stderr, "\n");
}

// ── on_load / on_unload ──────────────────────────────────────────────
static void on_load(void) {
    fprintf(stderr, "[fp8-kv] plugin loaded (fp8-kv-plugin v1.0)\n");
    const char* profile = getenv("SMART_KV_PROFILE");
    if (profile) g_profile_name = profile;
    detect_fp8_optin();
}

static void on_unload(void) {
    if (g_shared_ctx) {
        log_stats(g_shared_ctx);
        for (int64_t i = 0; i < g_shared_ctx->kv_size; i++) {
            free(g_shared_ctx->k_rows[i]);
            free(g_shared_ctx->v_rows[i]);
        }
        free(g_shared_ctx->k_rows);
        free(g_shared_ctx->v_rows);
        free(g_shared_ctx->slot_tier);
        free(g_shared_ctx->chunks);
        free(g_shared_ctx->total_writes);
        if (g_shared_ctx->train_ring) {
            ls_free_ring_buffer(g_shared_ctx->train_ring);
        }
        delete g_shared_ctx;
        g_shared_ctx = NULL;
    }
    fprintf(stderr, "[fp8-kv] plugin unloaded\n");
}

// ── KV cache store ───────────────────────────────────────────────────
static int kv_cache_store(void* layer_ctx, int64_t pos,
                           const void* k_data, const void* v_data,
                           int64_t n_tokens) {
    FP8KVContext* ctx = (FP8KVContext*)layer_ctx;
    if (!ctx || n_tokens <= 0) return 0;

    for (int64_t tok = 0; tok < n_tokens; tok++) {
        int64_t slot = (pos + tok) % ctx->kv_size;
        ctx->step++;

        // Update chunk metadata
        smart_kv_chunk_meta* c = &ctx->chunks[slot];
        c->pos_end = (uint32_t)(pos + tok + 1);
        if (c->n_tokens == 0) c->pos_begin = (uint32_t)(pos + tok);
        c->n_tokens += 1;
        c->last_used_step = ctx->step;
        c->access_count++;
        if (c->access_count > ctx->cached_max_access)
            ctx->cached_max_access = c->access_count;

        ctx->total_writes[slot]++;

        // Score the slot and get tier
        uint8_t tier = score_slot(ctx, slot);

        // Handle TQ6 CPU offload
        if (tier >= SMART_KV_TIER_ULTRA_TQ) {
            ctx->slot_tier[slot] = 6;

            // Allocate TQ rows if needed
            if (!ctx->k_rows[slot] && k_data) {
                ctx->k_rows[slot] = (uint8_t*)calloc(
                    (size_t)ctx->n_head_kv, ctx->tq_bytes);
                if (ctx->k_rows[slot]) ctx->n_tq_row_allocs++;
            }
            if (!ctx->v_rows[slot] && v_data) {
                ctx->v_rows[slot] = (uint8_t*)calloc(
                    (size_t)ctx->n_head_kv, ctx->tq_bytes);
                if (ctx->v_rows[slot]) ctx->n_tq_row_allocs++;
            }

            // Encode to TQ if we have data pointers
            if (k_data && ctx->k_rows[slot]) {
                for (int h = 0; h < ctx->n_head_kv; h++) {
                    const uint16_t* k_src = (const uint16_t*)k_data
                        + (uintptr_t)tok * ctx->n_embd_k_gqa
                        + (uintptr_t)h * ctx->head_dim_k;
                    encode_head_fp16(k_src, ctx->head_dim_k,
                                     ctx->k_rows[slot] + (size_t)h * ctx->tq_bytes,
                                     ctx->bits_per_angle);
                }
                ctx->n_tier6_stored++;
            }
            // If no data pointer (GPU buffer), skip encoding.
            // The TQ data will be populated on first CPU recall.
        } else {
            // Tiers 1-4: stored in the GPU tensor (F16 or FP8).
            // The ggml set_rows operation handles the conversion.
            // We just mark the slot as GPU-resident.
            ctx->cfg.memory_used++;
            ctx->slot_tier[slot] = 0;
        }
    }

    return 0;
}

// ── KV cache retrieve ────────────────────────────────────────────────
static int kv_cache_retrieve(void* layer_ctx, int64_t pos,
                              void* k_out, void* v_out,
                              int64_t n_tokens) {
    FP8KVContext* ctx = (FP8KVContext*)layer_ctx;
    if (!ctx || n_tokens <= 0) return 0;

    for (int64_t tok = 0; tok < n_tokens; tok++) {
        int64_t slot = (pos + tok) % ctx->kv_size;
        smart_kv_chunk_meta* c = &ctx->chunks[slot];

        c->last_used_step = ctx->step;
        c->access_count++;
        if (c->access_count > ctx->cached_max_access)
            ctx->cached_max_access = c->access_count;

        // If slot is TQ6, decode from TQ
        if (ctx->slot_tier[slot] >= 6 && ctx->k_rows[slot] && k_out) {
            for (int h = 0; h < ctx->n_head_kv; h++) {
                uint16_t* k_dst = (uint16_t*)k_out
                    + (uintptr_t)tok * ctx->n_embd_k_gqa
                    + (uintptr_t)h * ctx->head_dim_k;
                decode_head_fp16(
                    ctx->k_rows[slot] + (size_t)h * ctx->tq_bytes,
                    ctx->head_dim_k, k_dst, ctx->bits_per_angle);
            }
            ctx->n_tier6_retrieved++;
        }
        // For GPU-resident slots (tiers 1-4), the data is in the
        // ggml tensor. This function may not be called for those
        // slots (see PATCHES.md constraint #3).
    }

    return 0;
}

// ── KV cache clear ───────────────────────────────────────────────────
static void kv_cache_clear(void* layer_ctx) {
    FP8KVContext* ctx = (FP8KVContext*)layer_ctx;
    if (!ctx) return;

    for (int64_t i = 0; i < ctx->kv_size; i++) {
        free(ctx->k_rows[i]); ctx->k_rows[i] = NULL;
        free(ctx->v_rows[i]); ctx->v_rows[i] = NULL;
        ctx->slot_tier[i] = 0;
    }
    memset(ctx->chunks, 0, (size_t)ctx->kv_size * sizeof(smart_kv_chunk_meta));
    memset(ctx->total_writes, 0, (size_t)ctx->kv_size * sizeof(uint32_t));
    ctx->cfg.memory_used = 0;
    ctx->cached_max_access = 0;
    for (int t = 0; t <= SMART_KV_TIER_COUNT; t++)
        ctx->n_tier_decisions[t] = 0;
    ctx->n_total_tier_decisions = 0;
    ctx->n_tier6_stored = 0;
    ctx->n_tier6_retrieved = 0;
    ctx->step = 0;
}

// ── KV cache supported ───────────────────────────────────────────────
static bool kv_cache_supported(const ggml_plugin_kv_cache_params_t* params,
                                int bits_per_angle) {
    return params->head_dim_k > 0 && params->head_dim_v > 0;
}

// ── Plugin descriptor ────────────────────────────────────────────────
static const ggml_plugin_kv_cache_v1_t g_kv_cache_v1 = {
    .version            = 1,
    .on_load            = on_load,
    .on_model_loaded    = NULL,
    .on_unload          = on_unload,
    .kv_cache_supported = kv_cache_supported,
    .kv_cache_create_layer = kv_cache_create_layer,
    .kv_cache_destroy_layer = kv_cache_destroy_layer,
    .kv_cache_store     = kv_cache_store,
    .kv_cache_retrieve  = kv_cache_retrieve,
    .kv_cache_clear     = kv_cache_clear,
};

static const ggml_plugin_capability_t g_capabilities[] = {
    {
        .capability_id  = GGML_PLUGIN_CAP_KV_CACHE,
        .version        = 1,
        .function_table = (void*)&g_kv_cache_v1,
    },
};

static const ggml_plugin_descriptor_t g_desc = {
    .api_version        = GGML_PLUGIN_API_VERSION,
    .name               = "fp8-kv-plugin",
    .version            = "1.0.0",
    .description        = "FP8 GPU tensor + smart tiering + TQ6 CPU offload for ROCm KV cache",
    .num_capabilities   = 1,
    .capabilities       = g_capabilities,
};

extern "C" GGML_PLUGIN_EXPORT
ggml_plugin_descriptor_t* ggml_plugin_get_descriptor(void) {
    return &g_desc;
}
