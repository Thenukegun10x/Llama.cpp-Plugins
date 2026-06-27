// Smart-TQ Cache Plugin: smart scoring + TurboQuant CPU offload for tier 6
// Provides: GGML_PLUGIN_CAP_KV_CACHE (v1)
//
// Tiers 1-5: stored as F16 on GPU (standard path)
// Tier 6:    stored as TurboQuant on CPU only — VRAM-free
//
// The F16 GPU tensor is still allocated but tier-6 slots leave it ZEROED
// so it consumes only the fixed allocation, not per-slot memory pressure.
// True VRAM savings require the mixed-cache routing patch (see below).

#define _CRT_SECURE_NO_WARNINGS

#include "ggml-plugin.h"
#include "ggml.h"
#include "../turbo-quant-plugin/turboquant.h"  // PolarQuant+QJL
#include "smart-kv-cache.h"                     // scoring + tier assignment
#include "learned-score.h"                      // MLP scorers

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <new>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
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

// ── Per-layer context ────────────────────────────────────────────────

struct SmartTQContext {
    // Learned scorers (QuantRegressor + RAMDemoter)
    learned_scorer_t    lscorer;
    ls_ring_buffer_t*   train_ring;
    uint64_t            n_train_samples;

    // Model geometry
    int64_t n_embd_k_gqa;
    int64_t n_embd_v_gqa;
    int64_t kv_size;
    int32_t n_head_kv;
    int32_t head_dim_k;
    int32_t head_dim_v;
    int     bits_per_angle;

    // FP8 mode: when the GPU tensor is F8_E4M3, each slot uses 1 B/el instead of 2
    bool      fp8_active;      // true if tensor is F8_E4M3 (halved VRAM per slot)

    // TQ storage per slot, allocated dynamically when a slot enters tier 6.
    // Indirection: for each slot, store whether it's tier 6 (TQ only) or tier 1-5 (F16/FP8)
    uint8_t*  slot_tier;       // [kv_size] — 0 = GPU (F16 or FP8), 6 = Ultra-TQ
    uint8_t** k_rows;          // [kv_size] allocated only for tier-6 slots
    uint8_t** v_rows;          // [kv_size] allocated only for tier-6 slots
    size_t    tq_bytes;        // bytes per head vector in TQ format

    // Smart cache scoring state
    smart_kv_config   cfg;
    smart_kv_chunk_meta* chunks;  // [kv_size] — one per slot
    uint64_t          step;
    uint32_t*         total_writes;  // [kv_size] — persistent across cache clear

    // Cached values to avoid re-scanning
    uint32_t cached_max_access;

    // Stats
    uint64_t n_tier_decisions[SMART_KV_TIER_COUNT + 1];
    uint64_t n_total_tier_decisions;
    uint64_t n_tq_row_allocs;
    uint64_t n_tq_row_frees;
    uint64_t n_tier6_stored;
    uint64_t n_tier6_retrieved;
    uint64_t n_compress_events;
    uint64_t n_compress_slots;

    SmartTQContext() : fp8_active(false), n_train_samples(0), n_total_tier_decisions(0), n_tq_row_allocs(0), n_tq_row_frees(0), n_tier6_stored(0), n_tier6_retrieved(0), n_compress_events(0), n_compress_slots(0), cached_max_access(0), train_ring(NULL) {
        memset(n_tier_decisions, 0, sizeof(n_tier_decisions));
    }
};

// ── Lifecycle ────────────────────────────────────────────────────────

static bool g_loaded = false;
static bool g_train_mode = false;
static bool g_snapshot_exports = false;
static bool g_fp8_mode = false;
static bool g_compress_mode = false;
static const char* g_profile_name = "balanced";
static float g_offload_scale = 1.0f;
static uint64_t g_stats_interval = 8192;
static float g_ram_threshold = 0.80f;
static float g_compress_pressure = 0.80f;    // start compressing at 80% fill
static int   g_compress_block = 4;           // drop 4 slots at a time
static int   g_compress_sinks = 4;           // keep first N slots as sink anchors
static int   g_compress_interval = 256;      // check compression every N stores
static float g_compress_threshold = 0.70f;   // coldness threshold (0-1) to trigger

static void free_tq_slot(SmartTQContext* ctx, int64_t slot);
static void free_all_tq_slots(SmartTQContext* ctx);

// Singleton KV cache context — shared across all layers
static SmartTQContext* g_shared_ctx = NULL;
static char g_plugin_dir[512] = ".";
static char g_export_dir[512] = ".";

static void resolve_plugin_dir(void) {
#ifdef _WIN32
    HMODULE module = NULL;
    BOOL ok = GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)&resolve_plugin_dir,
        &module);
    if (ok && module) {
        char path[512];
        DWORD n = GetModuleFileNameA(module, path, (DWORD)sizeof(path));
        if (n > 0 && n < sizeof(path)) {
            char* slash = strrchr(path, '\\');
            char* fslash = strrchr(path, '/');
            if (!slash || (fslash && fslash > slash)) slash = fslash;
            if (slash) {
                *slash = '\0';
                snprintf(g_plugin_dir, sizeof(g_plugin_dir), "%s", path);
                return;
            }
        }
    }
#endif
    snprintf(g_plugin_dir, sizeof(g_plugin_dir), ".");
}

static void make_export_path(char* out, size_t out_size, const char* name) {
    if (!out || out_size == 0) return;
    const char sep =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif
    size_t n = strlen(g_export_dir);
    if (n > 0 && (g_export_dir[n - 1] == '\\' || g_export_dir[n - 1] == '/')) {
        snprintf(out, out_size, "%s%s", g_export_dir, name);
    } else {
        snprintf(out, out_size, "%s%c%s", g_export_dir, sep, name);
    }
}

static void on_load(void) {
    g_loaded = true;
    resolve_plugin_dir();
    const char* train_mode = getenv("SMART_KV_TRAIN_MODE");
    const char* snapshot_exports = getenv("SMART_KV_SNAPSHOT_EXPORTS");
    const char* export_dir = getenv("SMART_KV_EXPORT_DIR");
    const char* profile = getenv("SMART_KV_PROFILE");
    const char* fp8_env = getenv("SMART_KV_FP8");
    const char* stats_interval = getenv("SMART_KV_STATS_INTERVAL");
    const char* compress_env = getenv("SMART_KV_COMPRESS");
    const char* compress_pressure_env = getenv("SMART_KV_COMPRESS_PRESSURE");
    const char* compress_block_env = getenv("SMART_KV_COMPRESS_BLOCK");
    const char* compress_sinks_env = getenv("SMART_KV_COMPRESS_SINKS");
    const char* compress_interval_env = getenv("SMART_KV_COMPRESS_INTERVAL");
    const char* compress_threshold_env = getenv("SMART_KV_COMPRESS_THRESHOLD");
    g_fp8_mode = fp8_env && fp8_env[0] && strcmp(fp8_env, "0") != 0;
    g_compress_mode = compress_env && compress_env[0] && strcmp(compress_env, "0") != 0;
    if (compress_pressure_env && compress_pressure_env[0]) {
        float v = (float)atof(compress_pressure_env);
        if (v > 0.0f && v < 1.0f) g_compress_pressure = v;
    }
    if (compress_block_env && compress_block_env[0]) {
        int v = atoi(compress_block_env);
        if (v > 0 && v <= 256) g_compress_block = v;
    }
    if (compress_sinks_env && compress_sinks_env[0]) {
        int v = atoi(compress_sinks_env);
        if (v >= 0 && v <= 64) g_compress_sinks = v;
    }
    if (compress_interval_env && compress_interval_env[0]) {
        int v = atoi(compress_interval_env);
        if (v > 0) g_compress_interval = v;
    }
    if (compress_threshold_env && compress_threshold_env[0]) {
        float v = (float)atof(compress_threshold_env);
        if (v > 0.0f && v < 1.0f) g_compress_threshold = v;
    }
    g_train_mode = train_mode && train_mode[0] && strcmp(train_mode, "0") != 0;
    g_snapshot_exports = snapshot_exports && snapshot_exports[0] && strcmp(snapshot_exports, "0") != 0;
    if (stats_interval && stats_interval[0]) {
        g_stats_interval = strtoull(stats_interval, NULL, 10);
    }
    if (profile && profile[0]) {
        if (_stricmp(profile, "high") == 0 || _stricmp(profile, "very-high") == 0 || _stricmp(profile, "smart-high") == 0) {
            g_profile_name = "high";
            g_offload_scale = 0.50f;
            g_ram_threshold = 0.95f;
        } else if (_stricmp(profile, "balanced") == 0 || _stricmp(profile, "smart-balanced") == 0) {
            g_profile_name = "balanced";
            g_offload_scale = 1.00f;
            g_ram_threshold = 0.85f;
        } else if (_stricmp(profile, "perf") == 0 || _stricmp(profile, "performance") == 0 || _stricmp(profile, "smart-perf") == 0) {
            g_profile_name = "perf";
            g_offload_scale = 1.50f;
            g_ram_threshold = 0.75f;
        } else if (_stricmp(profile, "ultra") == 0 || _stricmp(profile, "smart-ultra") == 0) {
            g_profile_name = "ultra";
            g_offload_scale = 2.00f;
            g_ram_threshold = 0.65f;
        }
    }
    if (export_dir && export_dir[0]) {
        snprintf(g_export_dir, sizeof(g_export_dir), "%s", export_dir);
    } else {
        snprintf(g_export_dir, sizeof(g_export_dir), "%s", g_plugin_dir);
    }
    fprintf(stderr, "[smart-tq] Smart + TurboQuant cache plugin loaded\n");
    fprintf(stderr, "[smart-tq] plugin dir: %s\n", g_plugin_dir);
    fprintf(stderr, "[smart-tq] export dir: %s\n", g_export_dir);
    fprintf(stderr, "[smart-tq] profile: %s (offload scale %.2f, ram threshold %.2f)\n", g_profile_name, g_offload_scale, g_ram_threshold);
    fprintf(stderr, "[smart-tq] FP8 mode: %s (set SMART_KV_FP8=1 to enable)\n", g_fp8_mode ? "enabled" : "disabled");
    fprintf(stderr, "[smart-tq] live stats interval: %llu decisions\n", (unsigned long long)g_stats_interval);
    if (g_train_mode) {
        fprintf(stderr, "[smart-tq] training mode: heuristic teacher labels, learned weights disabled\n");
    }
    if (g_snapshot_exports) {
        fprintf(stderr, "[smart-tq] snapshot exports enabled\n");
    }
    if (g_compress_mode) {
        fprintf(stderr, "[smart-tq] context compression: enabled at %.0f%% fill, block=%d, sinks=%d, interval=%d, threshold=%.2f\n",
                g_compress_pressure * 100.0f, g_compress_block, g_compress_sinks, g_compress_interval, g_compress_threshold);
        fprintf(stderr, "[smart-tq]   tunables: SMART_KV_COMPRESS_PRESSURE=%.2f BLOCK=%d SINKS=%d INTERVAL=%d THRESHOLD=%.2f\n",
                g_compress_pressure, g_compress_block, g_compress_sinks, g_compress_interval, g_compress_threshold);
    }
}

static void dump_stats(SmartTQContext* ctx, const char* reason) {
    if (!ctx) return;

    uint64_t decisions = 0;
    for (int i = 1; i <= SMART_KV_TIER_COUNT; i++) decisions += ctx->n_tier_decisions[i];

    uint64_t occupied = ctx->cfg.memory_used;
    uint64_t tier6_slots = 0;
    uint64_t tq_bytes_live = 0;
    for (int64_t i = 0; i < ctx->kv_size; i++) {
        if (ctx->slot_tier[i] == SMART_KV_TIER_ULTRA_TQ) {
            tier6_slots++;
            if (ctx->k_rows[i]) tq_bytes_live += (uint64_t)ctx->n_head_kv * ctx->tq_bytes;
            if (ctx->v_rows[i]) tq_bytes_live += (uint64_t)ctx->n_head_kv * ctx->tq_bytes;
        }
    }

    fprintf(stderr, "[smart-tq] stats dump (%s): profile=%s fp8=%d decisions=%llu occupied=%llu/%lld tier6_slots=%llu tq_cpu=%.2f MiB allocs=%llu frees=%llu stored=%llu retrieved=%llu\n",
            reason ? reason : "manual",
            g_profile_name,
            ctx->fp8_active ? 1 : 0,
            (unsigned long long)decisions,
            (unsigned long long)occupied,
            (long long)ctx->kv_size,
            (unsigned long long)tier6_slots,
            (double)tq_bytes_live / (1024.0 * 1024.0),
            (unsigned long long)ctx->n_tq_row_allocs,
            (unsigned long long)ctx->n_tq_row_frees,
            (unsigned long long)ctx->n_tier6_stored,
            (unsigned long long)ctx->n_tier6_retrieved);

    for (int i = 1; i <= SMART_KV_TIER_COUNT; i++) {
        double pct = decisions ? (100.0 * (double)ctx->n_tier_decisions[i] / (double)decisions) : 0.0;
        fprintf(stderr, "[smart-tq]   tier %d %-15s decisions=%llu %.2f%%\n",
                i,
                smart_kv_tier_name((smart_kv_tier)i),
                (unsigned long long)ctx->n_tier_decisions[i],
                pct);
    }

    char path[512];
    make_export_path(path, sizeof(path), "smart_kv_stats.json");
    FILE* f = fopen(path, "w");
    if (f) {
        fprintf(f, "{\n");
        fprintf(f, "  \"profile\": \"%s\",\n", g_profile_name);
        fprintf(f, "  \"fp8_mode\": %d,\n", ctx->fp8_active ? 1 : 0);
        fprintf(f, "  \"reason\": \"%s\",\n", reason ? reason : "manual");
        fprintf(f, "  \"decisions\": %llu,\n", (unsigned long long)decisions);
        fprintf(f, "  \"occupied_slots\": %llu,\n", (unsigned long long)occupied);
        fprintf(f, "  \"kv_size\": %lld,\n", (long long)ctx->kv_size);
        fprintf(f, "  \"tier6_slots\": %llu,\n", (unsigned long long)tier6_slots);
        fprintf(f, "  \"tq_cpu_bytes_live\": %llu,\n", (unsigned long long)tq_bytes_live);
        fprintf(f, "  \"tq_row_allocs\": %llu,\n", (unsigned long long)ctx->n_tq_row_allocs);
        fprintf(f, "  \"tq_row_frees\": %llu,\n", (unsigned long long)ctx->n_tq_row_frees);
        fprintf(f, "  \"tier6_stored\": %llu,\n", (unsigned long long)ctx->n_tier6_stored);
        fprintf(f, "  \"tier6_retrieved\": %llu,\n", (unsigned long long)ctx->n_tier6_retrieved);
        fprintf(f, "  \"tiers\": [\n");
        for (int i = 1; i <= SMART_KV_TIER_COUNT; i++) {
            double pct = decisions ? (100.0 * (double)ctx->n_tier_decisions[i] / (double)decisions) : 0.0;
            fprintf(f, "    {\"tier\": %d, \"name\": \"%s\", \"decisions\": %llu, \"percent\": %.6f}%s\n",
                    i,
                    smart_kv_tier_name((smart_kv_tier)i),
                    (unsigned long long)ctx->n_tier_decisions[i],
                    pct,
                    i == SMART_KV_TIER_COUNT ? "" : ",");
        }
        fprintf(f, "  ]\n}\n");
        fclose(f);
        fprintf(stderr, "[smart-tq] wrote stats: %s\n", path);
    }
}

static void free_tq_slot(SmartTQContext* ctx, int64_t slot) {
    if (!ctx || slot < 0 || slot >= ctx->kv_size) return;
    if (ctx->k_rows && ctx->k_rows[slot]) {
        free(ctx->k_rows[slot]);
        ctx->k_rows[slot] = NULL;
        ctx->n_tq_row_frees++;
    }
    if (ctx->v_rows && ctx->v_rows[slot]) {
        free(ctx->v_rows[slot]);
        ctx->v_rows[slot] = NULL;
        ctx->n_tq_row_frees++;
    }
}

static void free_all_tq_slots(SmartTQContext* ctx) {
    if (!ctx || !ctx->k_rows || !ctx->v_rows) return;
    for (int64_t i = 0; i < ctx->kv_size; i++) {
        free_tq_slot(ctx, i);
    }
}

static void on_unload(void) {
    g_loaded = false;
    if (g_shared_ctx) {
        dump_stats(g_shared_ctx, "unload");
        free_all_tq_slots(g_shared_ctx);
        free(g_shared_ctx->k_rows);
        free(g_shared_ctx->v_rows);
        free(g_shared_ctx->slot_tier);
        free(g_shared_ctx->chunks);
        free(g_shared_ctx->total_writes);
        free(g_shared_ctx->train_ring);
        delete g_shared_ctx;
        g_shared_ctx = NULL;
    }
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
    if (!params) return false;
    if (bits_per_angle < 1 || bits_per_angle > 16) return false;
    if (!turboquant_valid_dims(params->head_dim_k)) return false;
    if (!turboquant_valid_dims(params->head_dim_v)) return false;
    return true;
}

// ── VRAM detection ──────────────────────────────────────────────────
#if defined(_WIN32) && !defined(__HIP__) && !defined(__CUDACC__)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static uint64_t detect_vram_budget(void) {
    size_t free = 0, total = 0;
    int ok = 0;
#if defined(__HIP__)
    ok = ((hipError_t)hipMemGetInfo(&free, &total) == hipSuccess);
#elif defined(__CUDACC__)
    ok = ((cudaError_t)cudaMemGetInfo(&free, &total) == cudaSuccess);
#else
    HMODULE h = LoadLibraryA("amdhip64.dll");
    if (!h) h = LoadLibraryA("hiprt.dll");
    if (h) {
        typedef int (*hip_mem_t)(size_t*, size_t*);
        auto fn = (hip_mem_t)GetProcAddress(h, "hipMemGetInfo");
        if (fn) ok = (fn(&free, &total) == 0);
        FreeLibrary(h);
    }
    if (!ok) {
        h = LoadLibraryA("nvcuda.dll");
        if (h) {
            typedef int (*cuda_mem_t)(size_t*, size_t*);
            auto fn = (cuda_mem_t)GetProcAddress(h, "cudaMemGetInfo_v2");
            if (!fn) fn = (cuda_mem_t)GetProcAddress(h, "cudaMemGetInfo");
            if (fn) ok = (fn(&free, &total) == 0);
            FreeLibrary(h);
        }
    }
#endif
    if (!ok) return 0;
    uint64_t budget = (uint64_t)((float)free * 0.85f);
    uint64_t overhead = 1073741824ull;
    return (budget > overhead) ? (budget - overhead) : 0;
}

// ── Layer create / destroy ──────────────────────────────────────────

static void* kv_cache_create_layer(const ggml_plugin_kv_cache_params_t* params,
                                    int bits_per_angle) {
    // Tier assignment is global per-position, not per-layer.
    // Return the same singleton for all layers to avoid duplicating
    // the ~10.9 MB TQ buffers per layer (32 layers × 2 K/V = 64 copies).
    // On second call (warmup), don't re-init — ring buffer data persists.
    if (g_shared_ctx) {
        // Mixed cache has buckets with different kv_sizes.
        // If this bucket is larger, grow allocations.
        if ((uint64_t)params->kv_size > g_shared_ctx->kv_size) {
            uint64_t old_kv = g_shared_ctx->kv_size;
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

            // Zero new entries
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

    SmartTQContext* ctx = new (std::nothrow) SmartTQContext();
    if (!ctx) return NULL;
    g_shared_ctx = ctx;

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

    // Allocate row-pointer tables only. Actual TQ CPU rows are allocated only
    // when a slot enters tier 6.
    ctx->k_rows = (uint8_t**)calloc((size_t)ctx->kv_size, sizeof(uint8_t*));
    ctx->v_rows = (uint8_t**)calloc((size_t)ctx->kv_size, sizeof(uint8_t*));
    ctx->slot_tier = (uint8_t*)calloc((size_t)ctx->kv_size, 1);

    // Allocate chunk metadata for smart scoring
    ctx->chunks = (smart_kv_chunk_meta*)calloc((size_t)ctx->kv_size,
                    sizeof(smart_kv_chunk_meta));
    ctx->total_writes = (uint32_t*)calloc((size_t)ctx->kv_size, sizeof(uint32_t));

    if (!ctx->k_rows || !ctx->v_rows || !ctx->slot_tier || !ctx->chunks || !ctx->total_writes) {
        free(ctx->k_rows); free(ctx->v_rows);
        free(ctx->slot_tier); free(ctx->chunks);
        delete ctx;
        return NULL;
    }

    // Initialize smart cache config
    ctx->cfg.weights = SMART_KV_WEIGHTS_NO_ATTN;
    if (strcmp(g_profile_name, "high") == 0) {
        ctx->cfg.weights = SMART_KV_TQ_UNITY_HIGH.weights;
    } else if (strcmp(g_profile_name, "balanced") == 0) {
        ctx->cfg.weights = SMART_KV_TQ_UNITY_BALANCED.weights;
    } else if (strcmp(g_profile_name, "perf") == 0) {
        ctx->cfg.weights = SMART_KV_TQ_UNITY_PERF.weights;
    } else if (strcmp(g_profile_name, "ultra") == 0) {
        ctx->cfg.weights = SMART_KV_TQ_UNITY_ULTRA.weights;
    }
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

    // Detect FP8 tensor support: user must set SMART_KV_FP8=1 AND use
    // --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 (requires ggml core patches).
    // The plugin detects the tensor type at init by checking params->type_k/type_v
    // or by env var opt-in. When FP8 is active, each GPU slot uses 1 B/el vs 2 B/el.
    ctx->fp8_active = g_fp8_mode;
    if (ctx->fp8_active) {
        fprintf(stderr, "[smart-tq] SMART_KV_FP8=1 — FP8 GPU tensor active (1 B/el, 2× VRAM efficiency)\n");
    }

    // Detect VRAM budget and set memory_capacity dynamically
    {
        uint64_t vram_budget = detect_vram_budget();
        if (vram_budget > 0) {
            // Bytes per slot: heads × max(head_dim) × 2 (K+V)
            // F16 = 2 bytes/el, FP8 = 1 byte/el
            uint64_t bytes_per_el = ctx->fp8_active ? 1 : 2;
            uint64_t bytes_per_slot = (uint64_t)ctx->n_head_kv
                * (uint64_t)(ctx->head_dim_k > ctx->head_dim_v ? ctx->head_dim_k : ctx->head_dim_v)
                * bytes_per_el * 2;  // K+V = 2 copies
            if (bytes_per_slot > 0) {
                uint64_t max_slots = vram_budget / bytes_per_slot;
                if (max_slots < ctx->kv_size) {
                    fprintf(stderr, "[smart-tq] VRAM budget: %llu MB, capping cache from %llu to %llu slots\n",
                            (unsigned long long)(vram_budget / (1024*1024)),
                            (unsigned long long)ctx->kv_size,
                            (unsigned long long)max_slots);
                    ctx->cfg.memory_capacity = (uint32_t)max_slots;
                } else {
                    ctx->cfg.memory_capacity = ctx->kv_size;
                }
            }
        }
        if (ctx->cfg.memory_capacity == 0) {
            ctx->cfg.memory_capacity = ctx->kv_size;  // default: full cache
        }
        // When FP8 is active, each GPU slot uses 1 B/el instead of 2 B/el,
        // so the same VRAM fits 2× more slots. Double effective capacity.
        if (ctx->fp8_active) {
            ctx->cfg.memory_capacity = (uint32_t)(ctx->cfg.memory_capacity * 2);
        }
    }

    // Initialize learned scorers + ring buffer
    memset(&ctx->lscorer, 0, sizeof(ctx->lscorer));
    ls_init_random(&ctx->lscorer, 42);
    if (g_train_mode || g_snapshot_exports) {
        ctx->train_ring = (ls_ring_buffer_t*)calloc(1, sizeof(ls_ring_buffer_t));
        if (ctx->train_ring) ls_ring_init(ctx->train_ring);
    }

    // Try to load pre-trained weights. Prefer files next to the loaded DLL;
    // fall back to repo-relative paths for manual launches from the repo root.
    // In training mode, leave quant_loaded/ram_loaded false so score_slot uses
    // the heuristic teacher instead of the previous learned model.
    if (!g_train_mode) {
        const char* dirs[] = {
            g_plugin_dir,
            "plugins/smart-kv-plugin",
            ".",
        };
        for (int i = 0; i < 3 && (!ctx->lscorer.quant_loaded || !ctx->lscorer.ram_loaded); i++) {
            ls_load_all_weights(&ctx->lscorer, dirs[i]);
        }
        if (ctx->lscorer.quant_loaded) {
            fprintf(stderr, "[smart-tq] QuantRegressor ready\n");
        } else {
            fprintf(stderr, "[smart-tq] QuantRegressor weights not found; using heuristic scoring\n");
        }
        if (ctx->lscorer.ram_loaded) {
            fprintf(stderr, "[smart-tq] RAMDemoter ready\n");
        } else {
            fprintf(stderr, "[smart-tq] RAMDemoter weights not found; RAM offload uses quant/pressure heuristic\n");
        }
    } else {
        fprintf(stderr, "[smart-tq] learned weights intentionally skipped for training collection\n");
    }

    size_t row_table_bytes = (size_t)ctx->kv_size * sizeof(uint8_t*) * 2;
    fprintf(stderr, "[smart-tq] layer %d: %ld slots x %d heads, "
            "TQ block=%zuB, dynamic row tables=%zuB\n",
            params->layer_idx, (long)ctx->kv_size, ctx->n_head_kv,
            ctx->tq_bytes, row_table_bytes);

    return ctx;
}

static void kv_cache_destroy_layer(void* layer_ctx) {
    // Singleton — real cleanup happens in on_unload.
    // llama.cpp calls destroy_layer per-layer (32×), which would
    // double-free the singleton if we freed here.
}

// ── VRAM detection ──────────────────────────────────────────────────
// Returns usable KV cache budget in bytes (85% free - 1 GB overhead).

#ifdef _WIN32
#include <windows.h>
#endif

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
    ctx->cached_max_access = max_access;

    // memory_used tracked incrementally in kv_cache_store and cache_clear

    // Quant scoring via QuantRegressor or heuristic fallback
    if (ctx->lscorer.quant_loaded) {
        ls_features_t feat = ls_features_from_meta(c, ctx->step, (uint32_t)ctx->kv_size, ctx->cfg.weights.gamma);
        float q_score = qr_forward(&ctx->lscorer.quant, &feat);
        float gamma = smart_kv_adaptive_gamma(ctx->cfg.weights.gamma, ctx->cfg.memory_used, ctx->cfg.memory_capacity);
        smart_kv_weights adj = ctx->cfg.weights;
        if (gamma != adj.gamma) {
            smart_kv_init_weights(&adj, gamma);
        }

        // Pressure scaling: smoothly lower quant tiers as VRAM fills
        // (operates independently of RAMDemoter which handles full CPU offload)
        float pressure = ctx->cfg.memory_capacity > 0
            ? (float)ctx->cfg.memory_used / (float)ctx->cfg.memory_capacity : 0.0f;
        if (pressure > 0.10f) {
            float factor = 1.0f - (pressure - 0.10f) * 0.5f;
            if (factor < 0.60f) factor = 0.60f;
            q_score *= factor;
        }

        // Dynamic RAM offload: if VRAM is nearly full and this chunk is cold,
        // evict to CPU via tier 6 instead of keeping on GPU at a lower quant.
        if (ctx->cfg.memory_capacity > 0 && !c->pinned) {
            if (ctx->lscorer.ram_loaded && pressure > 0.50f) {
                float ram_prob = rd_forward(&ctx->lscorer.ram, &feat, pressure);
                if (ram_prob >= g_ram_threshold) {
                    return SMART_KV_TIER_ULTRA_TQ;
                }
            }

            bool should_offload = false;
            float offload_threshold = 0.05f;
            if (pressure > 0.92f) offload_threshold = 0.20f;
            else if (pressure > 0.85f) offload_threshold = 0.12f;
            else if (pressure > 0.75f) offload_threshold = 0.10f;
            else if (pressure > 0.65f) offload_threshold = 0.08f;
            else if (pressure > 0.50f) offload_threshold = 0.05f;
            offload_threshold *= g_offload_scale;
            if (offload_threshold > 0.45f) offload_threshold = 0.45f;
            should_offload = (q_score <= offload_threshold && !c->pinned);
            if (should_offload) {
                return SMART_KV_TIER_ULTRA_TQ;
            }
        }

        smart_kv_tier t = smart_kv_assign_tier(q_score, c->min_tier, adj.priority_thresholds);
        return (uint8_t)t;
    }

    // Heuristic scoring (fallback)
    smart_kv_scored_chunk r = smart_kv_eval(c, ctx->step, max_access, &ctx->cfg);
    return (uint8_t)r.tier;
}

static uint64_t collect_training_sample(SmartTQContext* ctx, int64_t slot, uint32_t max_access) {
    if (!ctx || slot < 0 || slot >= ctx->kv_size) return ctx ? ctx->n_train_samples : 0;

    smart_kv_chunk_meta* c = &ctx->chunks[slot];
    if (c->access_count == 0) return ctx->n_train_samples;

    ls_features_t feat = ls_features_from_meta(c, ctx->step, (uint32_t)ctx->kv_size, ctx->cfg.weights.gamma);
    float mem_pressure = ctx->cfg.memory_capacity > 0
        ? (float)ctx->cfg.memory_used / (float)ctx->cfg.memory_capacity
        : 0.0f;

    smart_kv_scored_chunk teacher = smart_kv_eval(c, ctx->step, max_access, &ctx->cfg);
    uint8_t teacher_tier = (uint8_t)teacher.tier;

    float quant_label = 1.0f - ((float)(teacher_tier - 1) / (float)(SMART_KV_TIER_COUNT - 1));
    if (teacher_tier >= SMART_KV_TIER_ULTRA_TQ) quant_label = 0.0f;

    uint64_t age = ctx->step > c->last_used_step ? ctx->step - c->last_used_step : 0;
    bool low_priority = teacher.priority <= 0.28f;
    bool cold = age > (uint64_t)(ctx->kv_size / 16) || c->access_count <= 1;
    bool stale_rewrite = ctx->total_writes[slot] > 1 && c->access_count <= 1;
    bool pressure_candidate = smart_kv_should_offload(ctx->cfg.memory_used, ctx->cfg.memory_capacity, teacher.priority);

    float ram_label = 0.0f;
    uint8_t ht = teacher_tier;
    if (!c->pinned && low_priority && (cold || stale_rewrite || pressure_candidate)) {
        ram_label = 1.0f;
        ht = SMART_KV_TIER_ULTRA_TQ;
    }

    if (ctx->train_ring) {
        ls_ring_push(ctx->train_ring, &feat, quant_label, ram_label, mem_pressure,
                     c->chunk_id, c->access_count, ht);
    }
    return ++ctx->n_train_samples;
}

// ── Context compression ─────────────────────────────────────────────
// Drops the coldest contiguous block of slots to free up context space.
// Cold = low attention_ema + low access count + old last_used_step.
// Preserves sink tokens (first N) and recent window (last 25%).

static void compress_cold_slots(SmartTQContext* ctx) {
    if (!ctx || !g_compress_mode) return;
    if (ctx->kv_size < (int64_t)(g_compress_sinks + g_compress_block)) return;

    float pressure = ctx->cfg.memory_capacity > 0
        ? (float)ctx->cfg.memory_used / (float)ctx->cfg.memory_capacity : 0.0f;
    if (pressure < g_compress_pressure) return;

    // Find max access count for normalization
    uint32_t max_access = 1;
    for (int64_t i = g_compress_sinks; i < ctx->kv_size; i++) {
        if (ctx->chunks[i].access_count > max_access)
            max_access = ctx->chunks[i].access_count;
    }

    // Recent window: keep last 25% of slots
    int64_t recent_start = ctx->kv_size - ctx->kv_size / 4;
    if (recent_start < g_compress_sinks) recent_start = g_compress_sinks;

    // Score each slot by coldness: low attention + low access + old
    // Scan in the search region [g_compress_sinks, recent_start)
    int64_t search_start = g_compress_sinks;
    int64_t search_end   = recent_start;

    // Find the contiguous block with the highest average coldness
    int64_t best_start = search_start;
    double  best_cold  = -1.0;

    for (int64_t i = search_start; i <= search_end - g_compress_block; i++) {
        double total_cold = 0.0;
        for (int j = 0; j < g_compress_block; j++) {
            smart_kv_chunk_meta* c = &ctx->chunks[i + j];
            float norm_access = max_access > 0 ? (float)c->access_count / (float)max_access : 0.0f;
            total_cold += (1.0f - c->attention_ema) * (1.0f - norm_access);
        }
        double avg = total_cold / g_compress_block;
        if (avg > best_cold) {
            best_cold  = avg;
            best_start = i;
        }
    }

    // Only compress if cold enough (tunable via SMART_KV_COMPRESS_THRESHOLD)
    if (best_cold < g_compress_threshold) return;

    // Drop the block
    for (int j = 0; j < g_compress_block; j++) {
        int64_t slot = best_start + j;
        if (slot >= ctx->kv_size) break;

        free_tq_slot(ctx, slot);
        ctx->slot_tier[slot] = 0;
        if (ctx->chunks[slot].access_count > 0 && ctx->cfg.memory_used > 0) {
            ctx->cfg.memory_used--;
        }
        // Reset chunk metadata
        ctx->chunks[slot].access_count = 0;
        ctx->chunks[slot].attention_ema = 0.5f;
        ctx->chunks[slot].query_score = 0.5f;
        ctx->chunks[slot].k_norm = 0.0f;
        ctx->chunks[slot].v_norm = 0.0f;
        ctx->chunks[slot].k_variance = 0.0f;
        ctx->chunks[slot].n_tokens = 0;
    }

    ctx->n_compress_events++;
    ctx->n_compress_slots += g_compress_block;
    fprintf(stderr, "[smart-tq] compression event #%llu: freed %d slots at pos %lld (cold=%.3f, pressure=%.1f%%, total freed=%llu)\n",
            (unsigned long long)ctx->n_compress_events,
            g_compress_block, (long long)best_start, best_cold, pressure * 100.0f,
            (unsigned long long)ctx->n_compress_slots);
}

// ── kv_cache_store ──────────────────────────────────────────────────

static int kv_cache_store(void* layer_ctx, int64_t pos,
                           const void* k_data, const void* v_data,
                           int64_t n_tokens) {
    SmartTQContext* ctx = (SmartTQContext*)layer_ctx;
    if (!ctx) return -1;

    const uint16_t* k_fp16 = k_data ? (const uint16_t*)k_data : nullptr;
    const uint16_t* v_fp16 = v_data ? (const uint16_t*)v_data : nullptr;
    size_t row_bytes_k = (size_t)ctx->n_head_kv * ctx->tq_bytes;
    size_t row_bytes_v = (size_t)ctx->n_head_kv * ctx->tq_bytes;

    for (int64_t t = 0; t < n_tokens; t++) {
        int64_t slot = pos + t;
        if (slot < 0 || slot >= ctx->kv_size) return -1;

        ctx->step++;

        // Update persistent write counter (survives cache clear)
        ctx->total_writes[slot]++;

        // Snapshot pre-update state for coldness check
        smart_kv_chunk_meta* c = &ctx->chunks[slot];
        uint64_t prev_step = c->last_used_step;
        uint32_t prev_access = c->access_count;
        bool was_unoccupied = (c->access_count == 0 && ctx->slot_tier[slot] == 0);

        // Update chunk metadata
        c->last_used_step = ctx->step;
        c->access_count++;
        if (was_unoccupied) ctx->cfg.memory_used++;

        // Compute K/V norms if data is available (may be null for GPU-resident buffers)
        if (k_fp16 && v_fp16) {
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
                c->v_norm += (v_norm_tok - c->v_norm) / (float)(c->n_tokens + 1);
            }
            c->n_tokens++;
        }

        float attn = 0.5f;
        if (c->k_norm > 0.0f || c->v_norm > 0.0f) {
            float norm_signal = 0.5f * (c->k_norm + c->v_norm);
            attn = 0.35f + 0.30f * tanhf(norm_signal);
        }
        float lambda_a = ctx->cfg.weights.lambda_a;
        c->attention_ema = lambda_a * c->attention_ema + (1.0f - lambda_a) * attn;

        // Score to determine tier (uses QuantRegressor or heuristic)
        uint8_t tier = score_slot(ctx, slot);
        if (tier >= 1 && tier <= SMART_KV_TIER_COUNT) {
            ctx->n_tier_decisions[tier]++;
            ctx->n_total_tier_decisions++;
            if (g_stats_interval > 0 && (ctx->n_total_tier_decisions % g_stats_interval) == 0) {
                dump_stats(ctx, "live");
            }
        }

        // Context compression: periodically drop cold slots when cache is full
        if (g_compress_mode && (ctx->n_total_tier_decisions % g_compress_interval) == 0) {
            compress_cold_slots(ctx);
        }

        // Collect training data: sample current slot always, cold slots near write cursor
        if (g_train_mode || g_snapshot_exports) {
            uint32_t max_access = ctx->cached_max_access;

            uint64_t samples_seen = collect_training_sample(ctx, slot, max_access);
            int64_t kv_sz = (int64_t)ctx->kv_size;
            const int cold_offsets[] = { -256 };
            for (int c = 0; c < 1; c++) {
                int64_t cold = (slot + kv_sz + cold_offsets[c]) % kv_sz;
                int probes = 0;
                while ((cold == slot || ctx->chunks[cold].access_count == 0) && probes < (int)(kv_sz / 4)) {
                    cold = (cold + 1) % kv_sz;
                    probes++;
                }
                if (cold != slot && ctx->chunks[cold].access_count > 0)
                    samples_seen = collect_training_sample(ctx, cold, max_access);
            }

            // Optional debug snapshots. Disabled by default so training uses
            // one dataset accumulated over the whole collector run.
            if (ctx->train_ring && g_snapshot_exports &&
                (ctx->train_ring->count == 1000 || ctx->train_ring->count == 2000 ||
                 ctx->train_ring->count == 4000 || ctx->train_ring->count == 6000 ||
                 ctx->train_ring->count == 7500)) {
                // Debug: count positive RAM labels
                if (ctx->train_ring->count == 1000) {
                    int pos = 0;
                    for (uint32_t k = 0; k < ctx->train_ring->count; k++)
                        if (ctx->train_ring->samples[k].ram_label > 0.5f) pos++;
                    fprintf(stderr, "[debug-export] n=%u ram_pos=%u\n", ctx->train_ring->count, pos);
                }
                char name[64];
                char dpath[512];
                snprintf(name, sizeof(name), "training_data_%u.bin", ctx->train_ring->count);
                make_export_path(dpath, sizeof(dpath), name);
                int n = ls_export_dataset(dpath, ctx->train_ring);
                fprintf(stderr, "[smart-tq] exported %d samples to %s\n", n, dpath);
            }

            if (g_train_mode && samples_seen >= 1000 && (samples_seen % 4096) == 0) {
                char dpath[512];
                make_export_path(dpath, sizeof(dpath), "training_data_latest.bin");
                int n = ctx->train_ring ? ls_export_dataset(dpath, ctx->train_ring) : 0;
                uint32_t ram_pos = 0;
                if (ctx->train_ring) {
                    for (uint32_t k = 0; k < ctx->train_ring->count; k++)
                        if (ctx->train_ring->samples[k].ram_label > 0.5f) ram_pos++;
                }
                float pct = ctx->train_ring && ctx->train_ring->count > 0 ? (100.0f * ram_pos / ctx->train_ring->count) : 0.0f;
                fprintf(stderr, "[smart-tq] updated rolling dataset: %d samples (%llu seen) ram_pos=%u/%.1f%% to %s\n",
                        n, (unsigned long long)samples_seen, ram_pos, pct, dpath);
            }
        }

        if (tier == SMART_KV_TIER_ULTRA_TQ) {
            // ── Tier 6: TQ-encode on CPU only ──
            // k_fp16/v_fp16 may be NULL for GPU-resident buffers;
            // fall back to GPU F16 if data is not CPU-accessible.
            if (!k_fp16 || !v_fp16) {
                free_tq_slot(ctx, slot);
                ctx->slot_tier[slot] = 0;
                c->tier = 0;
                c->target_tier = 0;
                continue;
            }
            ctx->slot_tier[slot] = SMART_KV_TIER_ULTRA_TQ;
            if (!ctx->k_rows[slot]) {
                ctx->k_rows[slot] = (uint8_t*)malloc(row_bytes_k);
                ctx->n_tq_row_allocs++;
            }
            if (!ctx->v_rows[slot]) {
                ctx->v_rows[slot] = (uint8_t*)malloc(row_bytes_v);
                ctx->n_tq_row_allocs++;
            }
            if (!ctx->k_rows[slot] || !ctx->v_rows[slot]) {
                free_tq_slot(ctx, slot);
                ctx->slot_tier[slot] = 0;
                c->tier = 0;
                c->target_tier = 0;
                continue;
            }
            uint8_t* k_row = ctx->k_rows[slot];
            uint8_t* v_row = ctx->v_rows[slot];

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
            if (ctx->slot_tier[slot] == SMART_KV_TIER_ULTRA_TQ) {
                // Slot was previously TQ6 — now back on GPU, free CPU storage
                free_tq_slot(ctx, slot);
            }
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
        if (slot < 0 || slot >= ctx->kv_size) {
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
            c->promotion_count++;
            c->last_promotion_step = ctx->step;
            ctx->slot_tier[slot] = 0;
            c->tier = 0;
            c->target_tier = 0;

            uint8_t* k_row = ctx->k_rows[slot];
            uint8_t* v_row = ctx->v_rows[slot];
            if (!k_row || !v_row) {
                memset(k_fp16 + (size_t)t * ctx->n_embd_k_gqa, 0,
                       (size_t)ctx->n_embd_k_gqa * sizeof(uint16_t));
                memset(v_fp16 + (size_t)t * ctx->n_embd_v_gqa, 0,
                       (size_t)ctx->n_embd_v_gqa * sizeof(uint16_t));
                free_tq_slot(ctx, slot);
                continue;
            }

            for (int32_t h = 0; h < ctx->n_head_kv; h++) {
                int off_k = (int)((size_t)t * ctx->n_embd_k_gqa + (size_t)h * ctx->head_dim_k);
                int off_v = (int)((size_t)t * ctx->n_embd_v_gqa + (size_t)h * ctx->head_dim_v);
                decode_head_fp16(k_row + (size_t)h * ctx->tq_bytes, ctx->head_dim_k,
                                 k_fp16 + off_k, ctx->bits_per_angle);
                decode_head_fp16(v_row + (size_t)h * ctx->tq_bytes, ctx->head_dim_v,
                                 v_fp16 + off_v, ctx->bits_per_angle);
            }
            ctx->n_tier6_retrieved++;
            free_tq_slot(ctx, slot);

        } else {
            // Tier 1-5: data already in GPU F16 tensor — llama.cpp
            // fills k_out/v_out before this callback. No plugin action needed.
        }
    }

    return 0;
}

// ── kv_cache_clear ──────────────────────────────────────────────────

static void kv_cache_clear(void* layer_ctx) {
    SmartTQContext* ctx = (SmartTQContext*)layer_ctx;
    if (!ctx) return;

    // Export training data before clearing (with raw RAM labels)
    if (ctx->train_ring && ctx->train_ring->count > 100) {
        char name[64];
        char dpath[512];
        time_t now = time(NULL);
        struct tm * tm_info = localtime(&now);
        strftime(name, sizeof(name), "session_%Y%m%d_%H%M%S.bin", tm_info);
        make_export_path(dpath, sizeof(dpath), name);
        int n = ls_export_dataset(dpath, ctx->train_ring);
        fprintf(stderr, "[smart-tq] session ended, exported %d samples to %s\n", n, dpath);
    }

    dump_stats(ctx, "cache_clear");

    free_all_tq_slots(ctx);
    memset(ctx->slot_tier, 0, (size_t)ctx->kv_size);
    memset(ctx->chunks, 0, (size_t)ctx->kv_size * sizeof(smart_kv_chunk_meta));

    // Re-init chunk metadata
    for (int64_t i = 0; i < ctx->kv_size; i++) {
        ctx->chunks[i].chunk_id = (uint32_t)i;
        ctx->chunks[i].attention_ema = 0.5f;
        ctx->chunks[i].query_score = 0.5f;
    }

    ctx->step = 0;
    ctx->cfg.memory_used = 0;
    memset(ctx->n_tier_decisions, 0, sizeof(ctx->n_tier_decisions));
    ctx->n_total_tier_decisions = 0;
    ctx->n_tier6_stored = 0;
    ctx->n_tier6_retrieved = 0;

    // Re-init ring buffer for next session
    if (ctx->train_ring) ls_ring_init(ctx->train_ring);
    ctx->n_train_samples = 0;

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

// ── Tier routing export ──────────────────────────────────────────────
// Called by llama-model.cpp via ggml_plugin_get_export() to wire
// tier_fn_ so that tier-6 slots route to the TQ plugin bucket
// (kv_size=1 GPU cell) instead of the full F16 tensor.

extern "C" GGML_PLUGIN_EXPORT
uint8_t smart_tq_get_tier(uint32_t global_pos) {
    if (!g_shared_ctx) return 0;
    if ((int64_t)global_pos >= g_shared_ctx->kv_size) return 0;
    return g_shared_ctx->slot_tier[global_pos];
}
