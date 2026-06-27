#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Dual-model learned scorer ─────────────────────────────────────────
// Two independent tiny MLPs replacing the old single-score approach:
//
// QuantRegressor: assigns quant tier within GPU VRAM (tiers 1-5)
//   Input(23) → Linear(16) → ReLU → Linear(1) → Sigmoid → score
//   Params: 23*16 + 16 + 16*1 + 1 = 401
//
// RAMDemoter: decides whether to evict to CPU RAM (tier 6)
//   Input(24) → Linear(8) → ReLU → Linear(1) → Sigmoid → probability
//   Params: 24*8 + 8 + 8*1 + 1 = 209
//
// Combined: ~610 params, ~2.4 KB, ~1μs forward pass
//
// Input features (23 base):
//   recency_log, frequency_log, attention_ema, query_score,
//   tag_onehot[10], redundancy_score, chunk_age_ratio, pinned, gamma,
//   k_norm, v_norm, k_variance,
//   promotion_count_log, recently_promoted
//
// RAMDemoter gets 1 extra: memory_pressure

// ── QuantRegressor (tier 1-5) ─────────────────────────────────────────

#define QR_IN_DIM   23
#define QR_HID_DIM  16
#define QR_OUT_DIM  1

typedef struct {
    float w1[QR_IN_DIM * QR_HID_DIM];  // [23][16]
    float b1[QR_HID_DIM];              // [16]
    float w2[QR_HID_DIM * QR_OUT_DIM]; // [16][1]
    float b2[QR_OUT_DIM];              // [1]
} quant_regressor_t;

// ── RAMDemoter (tier 6 binary) ───────────────────────────────────────

#define RD_IN_DIM   24   // 23 base + memory_pressure
#define RD_HID_DIM  8
#define RD_OUT_DIM  1

typedef struct {
    float w1[RD_IN_DIM * RD_HID_DIM];  // [24][8]
    float b1[RD_HID_DIM];              // [8]
    float w2[RD_HID_DIM * RD_OUT_DIM]; // [8][1]
    float b2[RD_OUT_DIM];              // [1]
} ram_demoter_t;

// ── Combined container for convenience ────────────────────────────────

typedef struct {
    quant_regressor_t quant;
    ram_demoter_t     ram;
    bool              ram_loaded;   // true if ram weights successfully loaded
    bool              quant_loaded; // true if quant weights successfully loaded
} learned_scorer_t;

// ── Input features (23 base + 1 optional for RAM) ────────────────────

typedef struct {
    float recency_log;          // log(now - last_used + 1) / 16
    float frequency_log;        // log(access_count + 1) / 6
    float attention_ema;        // current attention EMA (0-1)
    float query_score;          // query match score (0-1) — stale if computed at write time
    float tag_onehot[10];       // 10 tags → 10 floats (0/1)
    float redundancy_score;     // (0-1)
    float chunk_age_ratio;      // pos_begin / context_len (0-1)
    float pinned;               // (0/1)
    float gamma;                // adaptive gamma (1.0-4.0, default 2.0)

    // K/V tensor statistics — computed during store, most predictive single features
    float k_norm;               // running mean L2 norm(K) / sqrt(head_dim)
    float v_norm;               // running mean L2 norm(V) / sqrt(head_dim)
    float k_variance;           // running variance of K norms across tokens in chunk

    // Promotion tracking — prevents tier-6 ping-pong
    float promotion_count_log;  // log(promotion_count + 1) / 4 (0-1, 0 = never promoted)
    float recently_promoted;    // 1.0 if promotion within last 256 steps, else 0.0
} ls_features_t;

// ── Forward: QuantRegressor ──────────────────────────────────────────
// Returns predicted score 0-1 → mapped to tiers 1-5 via thresholds

static inline float qr_forward(const quant_regressor_t * m, const ls_features_t * f) {
    float x[QR_IN_DIM];
    int idx = 0;
    x[idx++] = f->recency_log;
    x[idx++] = f->frequency_log;
    x[idx++] = f->attention_ema;
    x[idx++] = f->query_score;
    for (int t = 0; t < 10; t++) x[idx++] = f->tag_onehot[t];
    x[idx++] = f->redundancy_score;
    x[idx++] = f->chunk_age_ratio;
    x[idx++] = f->pinned;
    x[idx++] = f->gamma;
    x[idx++] = f->k_norm;
    x[idx++] = f->v_norm;
    x[idx++] = f->k_variance;
    x[idx++] = f->promotion_count_log;
    x[idx++] = f->recently_promoted;

    // Layer 1: h = ReLU(x W1 + b1)
    float h[QR_HID_DIM];
    for (int j = 0; j < QR_HID_DIM; j++) {
        float s = m->b1[j];
        for (int i = 0; i < QR_IN_DIM; i++)
            s += x[i] * m->w1[i * QR_HID_DIM + j];
        h[j] = s > 0.0f ? s : 0.0f;
    }

    // Layer 2: y = sigmoid(h W2 + b2)
    float s = m->b2[0];
    for (int j = 0; j < QR_HID_DIM; j++)
        s += h[j] * m->w2[j * QR_OUT_DIM + 0];
    float y = 1.0f / (1.0f + expf(-s));

    return y;
}

// ── Forward: RAMDemoter ──────────────────────────────────────────────
// Returns probability 0-1 (>0.5 → evict to CPU RAM)
// Extra feature: memory_pressure (0-1) appended after the 23 base features

static inline float rd_forward(const ram_demoter_t * m, const ls_features_t * f, float memory_pressure) {
    float x[RD_IN_DIM];
    int idx = 0;
    x[idx++] = f->recency_log;
    x[idx++] = f->frequency_log;
    x[idx++] = f->attention_ema;
    x[idx++] = f->query_score;
    for (int t = 0; t < 10; t++) x[idx++] = f->tag_onehot[t];
    x[idx++] = f->redundancy_score;
    x[idx++] = f->chunk_age_ratio;
    x[idx++] = f->pinned;
    x[idx++] = f->gamma;
    x[idx++] = f->k_norm;
    x[idx++] = f->v_norm;
    x[idx++] = f->k_variance;
    x[idx++] = f->promotion_count_log;
    x[idx++] = f->recently_promoted;
    x[idx++] = memory_pressure;  // 24th feature

    // Layer 1: h = ReLU(x W1 + b1)
    float h[RD_HID_DIM];
    for (int j = 0; j < RD_HID_DIM; j++) {
        float s = m->b1[j];
        for (int i = 0; i < RD_IN_DIM; i++)
            s += x[i] * m->w1[i * RD_HID_DIM + j];
        h[j] = s > 0.0f ? s : 0.0f;
    }

    // Layer 2: y = sigmoid(h W2 + b2)
    float s = m->b2[0];
    for (int j = 0; j < RD_HID_DIM; j++)
        s += h[j] * m->w2[j * RD_OUT_DIM + 0];
    float y = 1.0f / (1.0f + expf(-s));

    return y;
}

// ── Convenience: score quant + check RAM in one call ─────────────────
// Returns:
//   score: quant score 0-1 (for tier 1-5 mapping)
//   is_ram: true if RAMDemoter says evict (> threshold)
//   If RAM model not loaded, is_ram is always false and score decides all 6 tiers.

typedef struct {
    float score;
    bool  is_ram;
} ls_result_t;

static inline ls_result_t ls_eval(const learned_scorer_t * ls, const ls_features_t * f, float memory_pressure, float ram_threshold) {
    ls_result_t r;
    float q_score = ls->quant_loaded ? qr_forward(&ls->quant, f) : 0.5f;
    bool  is_ram = false;
    if (ls->ram_loaded) {
        float rd_prob = rd_forward(&ls->ram, f, memory_pressure);
        is_ram = rd_prob >= ram_threshold;
    }
    r.score = q_score;
    r.is_ram = is_ram;
    return r;
}

// ── Feature extraction from chunk meta ────────────────────────────────

#include "smart-kv-cache.h"

#define LS_PROMOTION_WINDOW 256

static inline ls_features_t ls_features_from_meta(const smart_kv_chunk_meta * c,
    uint64_t now, uint32_t context_len, float gamma)
{
    ls_features_t f;
    memset(&f, 0, sizeof(f));

    uint64_t age = (now > c->last_used_step) ? (now - c->last_used_step) : 0;
    f.recency_log = logf((float)age + 1.0f) / 16.0f;
    if (f.recency_log > 1.0f) f.recency_log = 1.0f;

    f.frequency_log = logf((float)(c->access_count + 1)) / 6.0f;
    if (f.frequency_log > 1.0f) f.frequency_log = 1.0f;

    f.attention_ema    = c->attention_ema;
    f.query_score      = c->query_score;
    f.redundancy_score = c->redundancy_score;
    f.chunk_age_ratio  = context_len > 0 ? (float)c->pos_begin / (float)context_len : 0.0f;
    f.pinned           = c->pinned ? 1.0f : 0.0f;
    f.gamma            = gamma;
    f.k_norm           = c->k_norm;
    f.v_norm           = c->v_norm;
    f.k_variance       = c->k_variance;

    f.promotion_count_log = logf((float)(c->promotion_count + 1)) / 4.0f;
    if (f.promotion_count_log > 1.0f) f.promotion_count_log = 1.0f;

    f.recently_promoted = (now - c->last_promotion_step) < LS_PROMOTION_WINDOW ? 1.0f : 0.0f;

    uint8_t tag = c->tag;
    if (tag < 10) f.tag_onehot[tag] = 1.0f;

    return f;
}

// ── Dataset records (for export → Python training) ────────────────────

typedef struct {
    ls_features_t features;
    float         quant_label;      // 0-1: target tier normalized (0.2 = tier 5, 1.0 = tier 1)
    float         ram_label;        // 0.0 = keep on GPU, 1.0 = evict to CPU
    float         memory_pressure;  // 0-1 at time of sample
    uint32_t      chunk_id;         // which chunk this sample belongs to
    uint32_t      snapshot_access;  // chunk's access_count when sample was taken
    uint8_t       heuristic_tier;   // tier assigned by heuristic scorer (ground truth)
} ls_sample_t;

// Ring buffer (shared — stores both labels per sample)
#define LS_RING_CAPACITY 131072  // ~2000 tokens before wrap; 13 MB

typedef struct {
    ls_sample_t samples[LS_RING_CAPACITY];
    uint32_t    head;
    uint32_t    count;
} ls_ring_buffer_t;

static inline void ls_ring_init(ls_ring_buffer_t * rb) {
    rb->head = 0;
    rb->count = 0;
}

static inline void ls_ring_push(ls_ring_buffer_t * rb, const ls_features_t * f,
    float quant_label, float ram_label, float memory_pressure,
    uint32_t chunk_id, uint32_t snapshot_access, uint8_t heuristic_tier)
{
    uint32_t i = rb->head;
    rb->samples[i].features = *f;
    rb->samples[i].quant_label = quant_label;
    rb->samples[i].ram_label = ram_label;
    rb->samples[i].memory_pressure = memory_pressure;
    rb->samples[i].chunk_id = chunk_id;
    rb->samples[i].snapshot_access = snapshot_access;
    rb->samples[i].heuristic_tier = heuristic_tier;
    rb->head = (i + 1) % LS_RING_CAPACITY;
    if (rb->count < LS_RING_CAPACITY) rb->count++;
}

// ── Fixup RAM labels using re-access counterfactual ───────────────────
// For each sample, check if the chunk was re-accessed after the sample was taken.
// If snapshot_access < current access_count → chunk was needed → ram_label = 0.0
// If snapshot_access == current access_count → chunk was stale → ram_label stays as set.
// chunks: array of smart_kv_chunk_meta (indexed by chunk_id)
// n_chunks: number of chunks in the array
// Call this BEFORE ls_export_dataset to get correct counterfactual labels.

static inline void ls_ring_fixup_labels(ls_ring_buffer_t * rb,
    const smart_kv_chunk_meta * chunks, uint32_t n_chunks)
{
    for (uint32_t i = 0; i < rb->count; i++) {
        ls_sample_t * s = &rb->samples[i];
        if (s->chunk_id < n_chunks) {
            uint32_t cur = chunks[s->chunk_id].access_count;
            // If chunk was re-accessed after this sample, it should NOT have been evicted
            if (cur > s->snapshot_access) {
                s->ram_label = 0.0f;
            }
        }
    }
}

// ── Export dataset for Python training ────────────────────────────────
// Format: [header: float n_samples, float n_features=23] [X: n × 23 floats]
//         [y_quant: n × 1 float] [y_ram: n × 1 float] [mem_pressure: n × 1 float]
//
// Call ls_ring_fixup_labels() first to get correct counterfactual RAM labels.

static inline int ls_export_dataset(const char * path, const ls_ring_buffer_t * rb) {
    FILE * fp = fopen(path, "wb");
    if (!fp) return -1;

    uint32_t n = rb->count;
    float header[2] = { (float)n, (float)QR_IN_DIM };
    fwrite(header, sizeof(float), 2, fp);

    // Features
    for (uint32_t i = 0; i < n; i++) {
        const ls_features_t * f = &rb->samples[i].features;
        float row[QR_IN_DIM];
        int idx = 0;
        row[idx++] = f->recency_log;
        row[idx++] = f->frequency_log;
        row[idx++] = f->attention_ema;
        row[idx++] = f->query_score;
        for (int t = 0; t < 10; t++) row[idx++] = f->tag_onehot[t];
        row[idx++] = f->redundancy_score;
        row[idx++] = f->chunk_age_ratio;
        row[idx++] = f->pinned;
        row[idx++] = f->gamma;
        row[idx++] = f->k_norm;
        row[idx++] = f->v_norm;
        row[idx++] = f->k_variance;
        row[idx++] = f->promotion_count_log;
        row[idx++] = f->recently_promoted;
        fwrite(row, sizeof(float), QR_IN_DIM, fp);
    }

    // Quant labels
    for (uint32_t i = 0; i < n; i++) {
        float label = rb->samples[i].quant_label;
        fwrite(&label, sizeof(float), 1, fp);
    }

    // RAM labels
    for (uint32_t i = 0; i < n; i++) {
        float label = rb->samples[i].ram_label;
        fwrite(&label, sizeof(float), 1, fp);
    }

    // Memory pressure
    for (uint32_t i = 0; i < n; i++) {
        float mp = rb->samples[i].memory_pressure;
        fwrite(&mp, sizeof(float), 1, fp);
    }

    fclose(fp);
    return (int)n;
}

// ── Load QuantRegressor weights from binary ───────────────────────────
// File format: [w1: 23*16=368 floats] [b1: 16 floats]
//              [w2: 16*1=16 floats] [b2: 1 float]

static inline int ls_load_quant_weights(const char * path, quant_regressor_t * m) {
    FILE * fp = fopen(path, "rb");
    if (!fp) return -1;

    size_t total = sizeof(m->w1) + sizeof(m->b1) + sizeof(m->w2) + sizeof(m->b2);
    size_t nread = 0;
    nread += fread(m->w1, 1, sizeof(m->w1), fp);
    nread += fread(m->b1, 1, sizeof(m->b1), fp);
    nread += fread(m->w2, 1, sizeof(m->w2), fp);
    nread += fread(m->b2, 1, sizeof(m->b2), fp);
    fclose(fp);

    return (nread == total) ? 0 : -1;
}

// ── Load RAMDemoter weights from binary ──────────────────────────────
// File format: [w1: 24*8=192 floats] [b1: 8 floats]
//              [w2: 8*1=8 floats] [b2: 1 float]

static inline int ls_load_ram_weights(const char * path, ram_demoter_t * m) {
    FILE * fp = fopen(path, "rb");
    if (!fp) return -1;

    size_t total = sizeof(m->w1) + sizeof(m->b1) + sizeof(m->w2) + sizeof(m->b2);
    size_t nread = 0;
    nread += fread(m->w1, 1, sizeof(m->w1), fp);
    nread += fread(m->b1, 1, sizeof(m->b1), fp);
    nread += fread(m->w2, 1, sizeof(m->w2), fp);
    nread += fread(m->b2, 1, sizeof(m->b2), fp);
    fclose(fp);

    return (nread == total) ? 0 : -1;
}

// ── Random init for bootstrapping ─────────────────────────────────────

static inline void qr_init_random(quant_regressor_t * m, unsigned int seed) {
    srand(seed);
    float scale1 = sqrtf(2.0f / QR_IN_DIM);
    float scale2 = sqrtf(2.0f / QR_HID_DIM);
    for (int i = 0; i < QR_IN_DIM * QR_HID_DIM; i++)
        m->w1[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale1;
    for (int i = 0; i < QR_HID_DIM; i++)
        m->b1[i] = 0.0f;
    for (int i = 0; i < QR_HID_DIM * QR_OUT_DIM; i++)
        m->w2[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale2;
    m->b2[0] = 0.0f;
}

static inline void rd_init_random(ram_demoter_t * m, unsigned int seed) {
    srand(seed + 1234);
    float scale1 = sqrtf(2.0f / RD_IN_DIM);
    float scale2 = sqrtf(2.0f / RD_HID_DIM);
    for (int i = 0; i < RD_IN_DIM * RD_HID_DIM; i++)
        m->w1[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale1;
    for (int i = 0; i < RD_HID_DIM; i++)
        m->b1[i] = 0.0f;
    for (int i = 0; i < RD_HID_DIM * RD_OUT_DIM; i++)
        m->w2[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale2;
    m->b2[0] = 0.0f;
}

static inline void ls_init_random(learned_scorer_t * ls, unsigned int seed) {
    qr_init_random(&ls->quant, seed);
    rd_init_random(&ls->ram, seed);
    ls->quant_loaded = false;
    ls->ram_loaded = false;
}

// ── Load all weights (convenience) ────────────────────────────────────
// Tries: quant_weights.bin + ram_weights.bin in the given directory.
// Leaves existing initialized weights untouched if files don't exist.
// loaded flags mean real files were loaded, not random fallback.

static inline int ls_load_all_weights(learned_scorer_t * ls, const char * dir) {
    char qpath[512], rpath[512];

    snprintf(qpath, sizeof(qpath), "%s/quant_weights.bin", dir);
    snprintf(rpath, sizeof(rpath), "%s/ram_weights.bin", dir);

    quant_regressor_t qtmp;
    ram_demoter_t rtmp;
    bool quant_ok = (ls_load_quant_weights(qpath, &qtmp) == 0);
    bool ram_ok   = (ls_load_ram_weights(rpath, &rtmp) == 0);

    if (quant_ok) {
        ls->quant = qtmp;
        ls->quant_loaded = true;
        fprintf(stderr, "[learned] loaded %s\n", qpath);
    }
    if (ram_ok) {
        ls->ram = rtmp;
        ls->ram_loaded = true;
        fprintf(stderr, "[learned] loaded %s\n", rpath);
    }

    return (quant_ok ? 1 : 0) + (ram_ok ? 1 : 0);
}

#ifdef __cplusplus
}
#endif
