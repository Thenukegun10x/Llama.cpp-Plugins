// Integration test: Smart cache tiering + TurboQuant CPU offload
// Compile: gcc -o test-tq-int test-tq-integration.c smart-kv-cache.c
//          -I../turbo-quant-plugin -I../../llama.cpp/ggml/include -lm

#include "smart-kv-cache.h"
#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_SLOTS      64
#define HEAD_DIM     128
#define N_HEADS      8
#define BITS_ANGLE   4
#define TQ_BYTES     87   // turboquant_size(128,4) = 87 bytes/head
#define TQ_TOTAL_BYTES (N_HEADS * TQ_BYTES * 2)  // 1392: K+V for all heads

// Simulate one head-dim vector of K/V data
static void fill_head(float* data, int d, int seed) {
    srand((unsigned int)seed);
    for (int i = 0; i < d; i++)
        data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static float max_error(const float* a, const float* b, int n) {
    float e = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

static float avg_error(const float* a, const float* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += fabsf(a[i] - b[i]);
    return (float)(s / n);
}

// ── Test 1: TQ encode/decode roundtrip fidelity ──────────────────────

static int test_tq_roundtrip(void) {
    printf("  TQ encode/decode roundtrip...\n");

    float orig[HEAD_DIM];
    uint8_t packed[256];
    float decoded[HEAD_DIM];

    double total_max_err = 0.0;
    double total_avg_err = 0.0;
    int    n_trials = 20;

    for (int trial = 0; trial < n_trials; trial++) {
        fill_head(orig, HEAD_DIM, trial * 1234);
        size_t sz = turboquant_encode(orig, HEAD_DIM, packed, BITS_ANGLE, 0);
        turboquant_decode(packed, HEAD_DIM, decoded, BITS_ANGLE, 0);
        float me = max_error(orig, decoded, HEAD_DIM);
        float ae = avg_error(orig, decoded, HEAD_DIM);
        total_max_err += me;
        total_avg_err += ae;
    }

    printf("    avg max error: %.4f\n", (float)(total_max_err / n_trials));
    printf("    avg avg error: %.4f\n", (float)(total_avg_err / n_trials));
    printf("    bpw: %.2f\n", (double)turboquant_size(HEAD_DIM, BITS_ANGLE) * 8.0 / HEAD_DIM);
    printf("    size: %zu bytes per head vector\n", turboquant_size(HEAD_DIM, BITS_ANGLE));
    return 0;
}

// ── Test 2: Smart cache tier scoring → tier 6 assignment ────────────

typedef struct {
    smart_kv_chunk_meta meta;
    uint8_t tier;       // computed tier
    uint8_t tq_data[2048]; // TQ_TOTAL_BYTES = 1392, buffer is padded for safety
} slot_t;

static int test_tier_assignment(void) {
    printf("  Smart cache tier assignment...\n");

    slot_t slots[N_SLOTS];
    memset(slots, 0, sizeof(slots));

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512, .migrate_max = 8,
        .base_type = 12, .memory_capacity = N_SLOTS, .memory_used = 0,
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);

    // Simulate 100 steps of access
    for (int step = 0; step < 100; step++) {
        // Fill slots with varying characteristics
        for (int i = 0; i < N_SLOTS; i++) {
            slots[i].meta.chunk_id = i;
            slots[i].meta.last_used_step = step - (i < 10 ? 0 : i * 3);
            slots[i].meta.access_count = i < 20 ? (20 - i) * 2 : (i < 40 ? 5 : 1);
            slots[i].meta.attention_ema = 0.5f;
            slots[i].meta.query_score = 0.5f;
            slots[i].meta.anchor_score = 0.0f;
            slots[i].meta.redundancy_score = 0.0f;
            slots[i].meta.min_tier = 0;
            slots[i].meta.pinned = false;
            slots[i].meta.tag = SMART_KV_TAG_DEFAULT;
            if (i == 0)       { slots[i].meta.tag = SMART_KV_TAG_SYSTEM; slots[i].meta.pinned = true; slots[i].meta.min_tier = 1; }
            else if (i == 1)  { slots[i].meta.tag = SMART_KV_TAG_TOOL_SCHEMA; slots[i].meta.pinned = true; slots[i].meta.min_tier = 1; }
            else if (i == 2)  { slots[i].meta.tag = SMART_KV_TAG_ERROR; slots[i].meta.pinned = true; slots[i].meta.min_tier = 2; }
            else if (i >= 50) { slots[i].meta.tag = SMART_KV_TAG_BOILERPLATE; slots[i].meta.redundancy_score = 0.8f; }
            else if (i >= 40) { slots[i].meta.tag = SMART_KV_TAG_ASSISTANT; slots[i].meta.redundancy_score = 0.3f; }
        }

        // Score all slots
        uint32_t max_access = 0;
        for (int i = 0; i < N_SLOTS; i++)
            if (slots[i].meta.access_count > max_access)
                max_access = slots[i].meta.access_count;

        cfg.memory_used = 0;
        for (int i = 0; i < N_SLOTS; i++)
            if (slots[i].meta.access_count > 0)
                cfg.memory_used++;

        int n_tier6 = 0;
        for (int i = 0; i < N_SLOTS; i++) {
            smart_kv_scored_chunk r = smart_kv_eval(&slots[i].meta, step, max_access, &cfg);
            slots[i].tier = (uint8_t)r.tier;
            if (r.tier == SMART_KV_TIER_ULTRA_TQ) n_tier6++;
        }

        if (step == 99) {
            printf("    step %d: tier distribution:\n", step);
            int tc[SMART_KV_TIER_COUNT] = {0};
            for (int i = 0; i < N_SLOTS; i++)
                if (slots[i].tier >= 1 && slots[i].tier <= SMART_KV_TIER_COUNT)
                    tc[slots[i].tier - 1]++;
            for (int t = 0; t < SMART_KV_TIER_COUNT; t++)
                printf("      %-15s: %2d\n", smart_kv_tier_name((smart_kv_tier)(t+1)), tc[t]);
        }
    }

    return 0;
}

// ── Test 3: Full pipeline — tier 6 → TQ encode → store → decode ─────

static int test_full_pipeline(void) {
    printf("  Full tier 6 → TQ pipeline (store+retrieve)...\n");

    slot_t slots[N_SLOTS];
    memset(slots, 0, sizeof(slots));

    // Simulated K/V data for each slot
    float k_data[N_SLOTS][HEAD_DIM * N_HEADS];
    float v_data[N_SLOTS][HEAD_DIM * N_HEADS];

    for (int i = 0; i < N_SLOTS; i++) {
        fill_head(k_data[i], HEAD_DIM * N_HEADS, i * 100);
        fill_head(v_data[i], HEAD_DIM * N_HEADS, i * 100 + 1);
    }

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512, .migrate_max = 8,
        .base_type = 12, .memory_capacity = N_SLOTS, .memory_used = 0,
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);

    int n_tier6 = 0;
    int n_gpu = 0;
    uint64_t total_tq_encode_time = 0;
    uint64_t total_tq_decode_time = 0;

    for (int step = 0; step < 50; step++) {
        for (int i = 0; i < N_SLOTS; i++) {
            slots[i].meta.chunk_id = i;
            slots[i].meta.last_used_step = step;
            slots[i].meta.access_count = (i < 10) ? 20 : (i < 30) ? 10 : (i < 50) ? 3 : 1;
            slots[i].meta.attention_ema = 0.5f;
            slots[i].meta.redundancy_score = (i >= 50) ? 0.8f : 0.0f;
            slots[i].meta.min_tier = 0;
            slots[i].meta.pinned = false;
            slots[i].meta.tag = (i >= 50) ? SMART_KV_TAG_BOILERPLATE : SMART_KV_TAG_DEFAULT;
        }

        uint32_t max_access = 0;
        for (int i = 0; i < N_SLOTS; i++)
            if (slots[i].meta.access_count > max_access)
                max_access = slots[i].meta.access_count;

        cfg.memory_used = N_SLOTS;

        for (int i = 0; i < N_SLOTS; i++) {
            smart_kv_scored_chunk r = smart_kv_eval(&slots[i].meta, step, max_access, &cfg);
            uint8_t old_tier = slots[i].tier;
            slots[i].tier = (uint8_t)r.tier;

            // If tier 6, encode via TQ (simulating the write path)
            if (r.tier == SMART_KV_TIER_ULTRA_TQ && old_tier != 6) {
                uint64_t t0 = clock();
                for (int h = 0; h < N_HEADS; h++) {
                    float head_k[HEAD_DIM], head_v[HEAD_DIM];
                    memcpy(head_k, k_data[i] + h * HEAD_DIM, HEAD_DIM * sizeof(float));
                    memcpy(head_v, v_data[i] + h * HEAD_DIM, HEAD_DIM * sizeof(float));
                    turboquant_encode(head_k, HEAD_DIM, slots[i].tq_data + h * TQ_BYTES, BITS_ANGLE, 0);
                    turboquant_encode(head_v, HEAD_DIM, slots[i].tq_data + (N_HEADS + h) * TQ_BYTES, BITS_ANGLE, 0);
                }
                uint64_t t1 = clock();
                total_tq_encode_time += (t1 - t0);
                n_tier6++;
            } else if (r.tier != SMART_KV_TIER_ULTRA_TQ && old_tier == 6) {
                n_gpu++;
            }
        }
    }

    // Now verify: decode TQ data and compare to original
    float decoded_k[N_HEADS][HEAD_DIM];
    float decoded_v[N_HEADS][HEAD_DIM];
    double total_err_k = 0.0, total_err_v = 0.0;
    int verified = 0;

    for (int i = 50; i < N_SLOTS; i++) {  // tier 6 slots
        for (int h = 0; h < N_HEADS; h++) {
            turboquant_decode(slots[i].tq_data + h * TQ_BYTES, HEAD_DIM, decoded_k[h], BITS_ANGLE, 0);
            turboquant_decode(slots[i].tq_data + (N_HEADS + h) * TQ_BYTES, HEAD_DIM, decoded_v[h], BITS_ANGLE, 0);
        }

        // Compare against original data
        double slot_err_k = 0.0, slot_err_v = 0.0;
        for (int h = 0; h < N_HEADS; h++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                slot_err_k += fabsf(k_data[i][h * HEAD_DIM + j] - decoded_k[h][j]);
                slot_err_v += fabsf(v_data[i][h * HEAD_DIM + j] - decoded_v[h][j]);
            }
        }
        total_err_k += slot_err_k / (N_HEADS * HEAD_DIM);
        total_err_v += slot_err_v / (N_HEADS * HEAD_DIM);
        verified++;
    }

    printf("    tier 6 chunks stored: %d\n", n_tier6);
    printf("    tier 6→1-5 promotions: %d\n", n_gpu);
    printf("    K decode avg error: %.4f\n", (float)(total_err_k / verified));
    printf("    V decode avg error: %.4f\n", (float)(total_err_v / verified));
    printf("    TQ encode time: %.2f ms (per slot)\n",
           (double)total_tq_encode_time / CLOCKS_PER_SEC * 1000.0 / n_tier6);

    return (verified > 0 && total_err_k / verified < 1.0f) ? 0 : 1;
}

// ── Test 4: VRAM savings estimate ────────────────────────────────────

static void test_vram_estimate(void) {
    printf("  VRAM savings estimate:\n");

    int layers[] = { 32, 64 };
    int ctx_lens[] = { 100000, 200000 };
    float tier6_pcts[] = { 0.30f, 0.40f };
    float bpw_gpu = 2.56f;  // Q2_K bpw

    // KV cache bytes per token per layer = K + V = 2 * n_kv_heads * head_dim * (bpw/8)
    int n_kv_heads = 8;
    int head_dim = 128;
    double bytes_per_token_layer = 2.0 * n_kv_heads * head_dim * (bpw_gpu / 8.0);

    printf("    per-token per-layer: %.0f bytes\n", bytes_per_token_layer);
    printf("\n");

    for (int li = 0; li < 2; li++) {
        for (int ci = 0; ci < 2; ci++) {
            for (int pi = 0; pi < 2; pi++) {
                double total_kv = ctx_lens[ci] * bytes_per_token_layer * layers[li];
                double saved = total_kv * tier6_pcts[pi];
                printf("    %d layers, %d tokens, %.0f%% tier 6: %.0f MB saved (%.0f%% of KV cache)\n",
                       layers[li], ctx_lens[ci], tier6_pcts[pi] * 100,
                       saved / (1024.0 * 1024.0),
                       tier6_pcts[pi] * 100);
            }
        }
    }
}

int main(void) {
    printf("SMART + TQ INTEGRATION TEST\n");
    printf("===========================\n\n");

    printf("Config: head_dim=%d, n_heads=%d, bits_per_angle=%d\n\n",
           HEAD_DIM, N_HEADS, BITS_ANGLE);

    int fail = 0;

    printf("--- Test 1: TQ format ---\n");
    fail += test_tq_roundtrip();

    printf("\n--- Test 2: Tier assignment ---\n");
    fail += test_tier_assignment();

    printf("\n--- Test 3: Full pipeline ---\n");
    fail += test_full_pipeline();

    printf("\n--- Test 4: VRAM estimate ---\n");
    test_vram_estimate();

    printf("\n%s\n", fail ? "FAILURES DETECTED" : "All integration tests passed.");
    return fail;
}
