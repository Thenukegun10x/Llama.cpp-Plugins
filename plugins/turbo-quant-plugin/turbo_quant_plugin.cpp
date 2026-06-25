#include "ggml-plugin.h"
#include "turboquant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>

/* ── FP16 conversion (duplicated from turboquant.cpp for linker isolation) ── */

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

/* ── Per-layer context ───────────────────────────────────────────────── */

struct LayerContext {
    int64_t  n_embd_k_gqa;
    int64_t  n_embd_v_gqa;
    int64_t  kv_size;
    int32_t  n_head_kv;
    int32_t  head_dim_k;
    int32_t  head_dim_v;
    int      bits_per_angle;
    size_t   elem_bytes;    /* bytes per encoded (head_dim) vector */
    uint8_t* k_data;        /* [kv_size][n_head_kv][elem_bytes]    */
    uint8_t* v_data;        /* [kv_size][n_head_kv][elem_bytes]    */
    bool*    valid;         /* [kv_size] has slot been written?    */
};

static size_t layer_kv_bytes(const LayerContext* ctx) {
    return (size_t)ctx->kv_size * (size_t)ctx->n_head_kv * ctx->elem_bytes;
}

/* ── Lifecycle ───────────────────────────────────────────────────────── */

static bool g_loaded = false;

static void on_load(void) {
    g_loaded = true;
    fprintf(stderr, "[turbo-quant] loaded (PolarQuant+QJL)\n");
}

static void on_unload(void) {
    g_loaded = false;
    fprintf(stderr, "[turbo-quant] unloaded\n");
}

static void on_model_loaded(const ggml_plugin_model_info_t* info,
                            size_t* scratch_size_out) {
    fprintf(stderr, "[turbo-quant] model: %ld ctx, %ld layers, hd=%ld\n",
            (long)info->n_ctx_train, (long)info->n_layers,
            (long)info->head_dim);
    *scratch_size_out = 0;
}

/* ── kv_cache_supported ──────────────────────────────────────────────── */

static bool kv_cache_supported(const ggml_plugin_kv_cache_params_t* params,
                                int bits_per_angle) {
    if (bits_per_angle < 1 || bits_per_angle > 16) return false;
    if (!turboquant_valid_dims(params->head_dim_k)) return false;
    if (!turboquant_valid_dims(params->head_dim_v)) return false;
    return true;
}

/* ── Layer create / destroy ──────────────────────────────────────────── */

static void* kv_cache_create_layer(const ggml_plugin_kv_cache_params_t* params,
                                    int bits_per_angle) {
    LayerContext* ctx = new (std::nothrow) LayerContext();
    if (!ctx) return NULL;

    ctx->n_embd_k_gqa  = params->n_embd_k_gqa;
    ctx->n_embd_v_gqa  = params->n_embd_v_gqa;
    ctx->kv_size       = params->kv_size;
    ctx->n_head_kv     = params->n_head_kv;
    ctx->head_dim_k    = params->head_dim_k;
    ctx->head_dim_v    = params->head_dim_v;
    ctx->bits_per_angle = bits_per_angle;

    int hd = (params->head_dim_k > params->head_dim_v)
             ? params->head_dim_k : params->head_dim_v;
    ctx->elem_bytes = turboquant_size(hd, bits_per_angle);

    size_t kb = layer_kv_bytes(ctx);
    ctx->k_data = (uint8_t*)malloc(kb);
    ctx->v_data = (uint8_t*)malloc(kb);
    ctx->valid  = (bool*)calloc((size_t)ctx->kv_size, sizeof(bool));

    if (!ctx->k_data || !ctx->v_data || !ctx->valid) {
        free(ctx->k_data); free(ctx->v_data); free(ctx->valid);
        delete ctx;
        return NULL;
    }

    memset(ctx->k_data, 0, kb);
    memset(ctx->v_data, 0, kb);

    fprintf(stderr, "[turbo-quant] layer %d: %ld slots x %d heads x %zuB = %.1f MB\n",
            params->layer_idx, (long)ctx->kv_size, ctx->n_head_kv,
            ctx->elem_bytes,
            (kb * 2.0f) / (1024.0f * 1024.0f));

    return ctx;
}

static void kv_cache_destroy_layer(void* layer_ctx) {
    if (!layer_ctx) return;
    LayerContext* ctx = (LayerContext*)layer_ctx;
    free(ctx->k_data);
    free(ctx->v_data);
    free(ctx->valid);
    delete ctx;
}

/* ── Convert one head's FP16 data to float, encode with TurboQuant ──── */

static void encode_head_fp16(const uint16_t* fp16_src, int head_dim,
                              uint8_t* dst, int bits)
{
    float* tmp = (float*)malloc((size_t)head_dim * sizeof(float));
    for (int i = 0; i < head_dim; i++) {
        tmp[i] = f16_to_f32(fp16_src[i]);
    }
    turboquant_encode(tmp, head_dim, dst, bits, 0);
    free(tmp);
}

static void decode_head_fp16(const uint8_t* src, int head_dim,
                              uint16_t* fp16_dst, int bits)
{
    float* tmp = (float*)malloc((size_t)head_dim * sizeof(float));
    turboquant_decode(src, head_dim, tmp, bits, 0);
    for (int i = 0; i < head_dim; i++) {
        fp16_dst[i] = f32_to_f16(tmp[i]);
    }
    free(tmp);
}

/* ── kv_cache_store ──────────────────────────────────────────────────── */

static int kv_cache_store(void* layer_ctx, int64_t pos,
                           const void* k_data, const void* v_data,
                           int64_t n_tokens)
{
    LayerContext* ctx = (LayerContext*)layer_ctx;
    if (!ctx) return -1;

    const uint16_t* k_fp16 = (const uint16_t*)k_data;
    const uint16_t* v_fp16 = (const uint16_t*)v_data;

    size_t row_bytes_k = ctx->n_head_kv * ctx->elem_bytes;
    size_t row_bytes_v = ctx->n_head_kv * ctx->elem_bytes;

    for (int64_t t = 0; t < n_tokens; t++) {
        int64_t slot = pos + t;
        if (slot >= ctx->kv_size) return -1;

        uint8_t* k_row = ctx->k_data + slot * row_bytes_k;
        uint8_t* v_row = ctx->v_data + slot * row_bytes_v;

        for (int32_t h = 0; h < ctx->n_head_kv; h++) {
            int hd_k = ctx->head_dim_k;
            int hd_v = ctx->head_dim_v;

            encode_head_fp16(k_fp16 + (size_t)t * ctx->n_embd_k_gqa + (size_t)h * hd_k,
                             hd_k, k_row + (size_t)h * ctx->elem_bytes,
                             ctx->bits_per_angle);

            encode_head_fp16(v_fp16 + (size_t)t * ctx->n_embd_v_gqa + (size_t)h * hd_v,
                             hd_v, v_row + (size_t)h * ctx->elem_bytes,
                             ctx->bits_per_angle);
        }

        ctx->valid[slot] = true;
    }

    return 0;
}

/* ── kv_cache_retrieve ───────────────────────────────────────────────── */

static int kv_cache_retrieve(void* layer_ctx, int64_t pos,
                              void* k_out, void* v_out,
                              int64_t n_tokens)
{
    LayerContext* ctx = (LayerContext*)layer_ctx;
    if (!ctx) return -1;

    uint16_t* k_fp16 = (uint16_t*)k_out;
    uint16_t* v_fp16 = (uint16_t*)v_out;

    size_t row_bytes_k = ctx->n_head_kv * ctx->elem_bytes;
    size_t row_bytes_v = ctx->n_head_kv * ctx->elem_bytes;

    for (int64_t t = 0; t < n_tokens; t++) {
        int64_t slot = pos + t;
        if (slot >= ctx->kv_size) return -1;
        if (!ctx->valid[slot]) {
            memset(k_fp16 + (size_t)t * ctx->n_embd_k_gqa, 0,
                   (size_t)ctx->n_embd_k_gqa * sizeof(uint16_t));
            memset(v_fp16 + (size_t)t * ctx->n_embd_v_gqa, 0,
                   (size_t)ctx->n_embd_v_gqa * sizeof(uint16_t));
            continue;
        }

        uint8_t* k_row = ctx->k_data + slot * row_bytes_k;
        uint8_t* v_row = ctx->v_data + slot * row_bytes_v;

        for (int32_t h = 0; h < ctx->n_head_kv; h++) {
            int hd_k = ctx->head_dim_k;
            int hd_v = ctx->head_dim_v;

            decode_head_fp16(k_row + (size_t)h * ctx->elem_bytes, hd_k,
                             k_fp16 + (size_t)t * ctx->n_embd_k_gqa + (size_t)h * hd_k,
                             ctx->bits_per_angle);

            decode_head_fp16(v_row + (size_t)h * ctx->elem_bytes, hd_v,
                             v_fp16 + (size_t)t * ctx->n_embd_v_gqa + (size_t)h * hd_v,
                             ctx->bits_per_angle);
        }
    }

    return 0;
}

/* ── kv_cache_clear ──────────────────────────────────────────────────── */

static void kv_cache_clear(void* layer_ctx) {
    LayerContext* ctx = (LayerContext*)layer_ctx;
    if (!ctx) return;
    memset(ctx->valid, 0, (size_t)ctx->kv_size * sizeof(bool));
    memset(ctx->k_data, 0, layer_kv_bytes(ctx));
    memset(ctx->v_data, 0, layer_kv_bytes(ctx));
}

/* ═══════════════════════════════════════════════════════════════════════
 * CAPABILITY DECLARATION
 * ═══════════════════════════════════════════════════════════════════════ */

static ggml_plugin_kv_cache_v1_t g_kv_cache_v1 = {
    .version             = 1,
    .on_load             = on_load,
    .on_model_loaded     = on_model_loaded,
    .on_unload           = on_unload,
    .kv_cache_supported  = kv_cache_supported,
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
    .name             = "turbo-quant",
    .version          = "1.0.0",
    .description      = "PolarQuant+QJL KV cache compression (Google TurboQuant-style)",
    .num_capabilities = 1,
    .capabilities     = g_caps,
};

extern "C" GGML_PLUGIN_EXPORT
ggml_plugin_descriptor_t* ggml_plugin_get_descriptor(void) {
    return &g_desc;
}
