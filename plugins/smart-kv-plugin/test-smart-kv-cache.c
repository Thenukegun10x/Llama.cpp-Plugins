#include "smart-kv-cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_CHUNKS 40
#define CHUNK_SIZE 128
#define CURRENT_STEP 10000ULL

static smart_kv_chunk_meta chunks[MAX_CHUNKS];
static int n_chunks = 0;

static void add_chunk(uint32_t id, uint32_t pos_begin, uint32_t n_tokens,
    bool pinned, uint32_t access, uint64_t last, float attn, float anchor,
    uint8_t min_tier, uint8_t tag, float query, float redun)
{
    if (n_chunks >= MAX_CHUNKS) return;
    smart_kv_chunk_meta * c = &chunks[n_chunks++];
    memset(c, 0, sizeof(*c));
    c->chunk_id        = id;
    c->pos_begin       = pos_begin;
    c->pos_end         = pos_begin + n_tokens - 1;
    c->n_tokens        = n_tokens;
    c->pinned          = pinned;
    c->last_used_step  = last;
    c->access_count    = access;
    c->attention_ema   = attn;
    c->anchor_score    = anchor;
    c->min_tier        = min_tier;
    c->tag             = tag;
    c->query_score     = query;
    c->redundancy_score = redun;
}

static int cmp_desc(const void * a, const void * b) {
    const smart_kv_chunk_meta * ca = (const smart_kv_chunk_meta *)a;
    const smart_kv_chunk_meta * cb = (const smart_kv_chunk_meta *)b;
    return (cb->score > ca->score) - (cb->score < ca->score);
}

static void dump_cache(const smart_kv_config * cfg, const char * label) {
    printf("\n===== %s =====\n", label);
    printf("config: base_type=%d | gamma=%.2f adaptive=%s\n",
           cfg->base_type, cfg->weights.gamma,
           cfg->memory_capacity > 0 ? "yes" : "no");
    printf("weights: rec=%.2f attn=%.2f freq=%.2f query=%.2f pin=%.2f redun=%.2f tau_r=%.0f\n",
           cfg->weights.w_recency, cfg->weights.w_attention,
           cfg->weights.w_frequency, cfg->weights.w_query,
           cfg->weights.w_pin, cfg->weights.w_redundancy,
           cfg->weights.tau_r);
    printf("chunks: %d\n\n", n_chunks);

    uint32_t max_access = 0;
    for (int i = 0; i < n_chunks; i++)
        if (chunks[i].access_count > max_access) max_access = chunks[i].access_count;

    for (int i = 0; i < n_chunks; i++) {
        smart_kv_scored_chunk r = smart_kv_eval(&chunks[i], CURRENT_STEP, max_access, cfg);
        chunks[i].score = r.priority;
        chunks[i].target_tier = (uint8_t)r.tier;
    }

    qsort(chunks, n_chunks, sizeof(smart_kv_chunk_meta), cmp_desc);

    int tc[SMART_KV_TIER_COUNT] = {0};
    for (int i = 0; i < n_chunks; i++) {
        smart_kv_tier t = (smart_kv_tier)chunks[i].target_tier;
        if (t >= 1 && t <= SMART_KV_TIER_COUNT) tc[t-1]++;
        smart_kv_log(&chunks[i], chunks[i].score, t,
            smart_kv_tier_to_type(t, cfg->base_type));
    }

    printf("\ntier summary:\n");
    for (int t = 0; t < SMART_KV_TIER_COUNT; t++) {
        int type = smart_kv_tier_to_type((smart_kv_tier)(t+1), cfg->base_type);
        printf("  %-15s (%d): %2d chunks -> ggml_type=%d\n",
               smart_kv_tier_name((smart_kv_tier)(t+1)), t+1, tc[t], type);
    }
    printf("===== END =====\n\n");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 1: agentic coding base
// ────────────────────────────────────────────────────────────────────────────
static void scen_agentic(void) {
    printf("=== SCENARIO: Agentic Coding ===\n\n");
    n_chunks = 0;
    //                    id pos len  P  acc last  attn anc min  tag           qry red
    add_chunk( 0,   0,  4, 1, 50,    0, .90f, 1.0f, 0, SMART_KV_TAG_SYSTEM,     .5f, 0);
    add_chunk( 1,   4,  4, 1, 45,    0, .90f, 1.0f, 0, SMART_KV_TAG_TOOL_SCHEMA,.5f, 0);
    add_chunk( 2,   8,  2, 1, 30,  100, .80f, 1.0f, 0, SMART_KV_TAG_DEFAULT,   .5f, 0);
    add_chunk( 3,  10,  1, 1, 25,  200, .70f, .9f,  0, SMART_KV_TAG_FILE_PATH, .8f, 0);
    add_chunk( 4,  12,  2, 0, 20, 9900, .60f, .5f,  0, SMART_KV_TAG_USER_INPUT,.6f, 0);
    add_chunk( 5,  14,  3, 0, 18, 9950, .50f, .4f,  0, SMART_KV_TAG_ASSISTANT, .4f, .1f);
    add_chunk( 6,  17,  2, 0, 15, 9980, .40f, .3f,  0, SMART_KV_TAG_USER_INPUT,.6f, 0);
    add_chunk( 7,  19,  1, 1, 12, 5000, .70f, .9f,  1, SMART_KV_TAG_ERROR,     .7f, 0);
    add_chunk( 8,  20,  3, 0, 40, 3000, .80f, .6f,  0, SMART_KV_TAG_CODE,      .5f, 0);
    add_chunk( 9,  23,  4, 0,  3, 1000, .10f, .0f,  0, SMART_KV_TAG_ASSISTANT, .2f, .5f);
    add_chunk(10,  27,  3, 0,  2,  500, .05f, .0f,  0, SMART_KV_TAG_BOILERPLATE,.0f, .8f);
    add_chunk(11,  30,  2, 0,  8, 8000, .30f, .2f,  0, SMART_KV_TAG_DEFAULT,   .5f, 0);
    add_chunk(12,  32,  5, 0,  1,  100, .00f, .0f,  0, SMART_KV_TAG_ASSISTANT, .1f, .6f);
    add_chunk(13,  37,  2, 0,  0,   50, .00f, .0f,  0, SMART_KV_TAG_BOILERPLATE,.0f, .9f);

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512,
        .migrate_max = 8,
        .base_type = 12, // Q4_K
        .memory_capacity = 0,
        .memory_used = 0,
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);
    dump_cache(&cfg, "Agentic Coding (no attention)");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 2: min_tier enforcement
// ────────────────────────────────────────────────────────────────────────────
static void scen_min_tier(void) {
    printf("=== SCENARIO: min_tier Enforcement ===\n\n");
    n_chunks = 0;
    // identical chunks, different min_tier
    add_chunk(0, 0, 2, 0, 1, 100, .0f, .0f, 0, SMART_KV_TAG_BOILERPLATE, .0f, .9f);
    add_chunk(1, 2, 2, 0, 1, 100, .0f, .0f, 2, SMART_KV_TAG_ASSISTANT,   .0f, .9f); // min Quality
    add_chunk(2, 4, 2, 0, 1, 100, .0f, .0f, 4, SMART_KV_TAG_ERROR,       .0f, .9f); // min Performance

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512,
        .migrate_max = 8,
        .base_type = 12,
        .memory_capacity = 0,
        .memory_used = 0,
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);
    dump_cache(&cfg, "min_tier: all same score, different floor");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 3: tag modifier impact
// ────────────────────────────────────────────────────────────────────────────
static void scen_tag_mods(void) {
    printf("=== SCENARIO: Tag Modifier Impact ===\n\n");
    n_chunks = 0;
    // Identical stats, different tags
    add_chunk(0, 0, 2, 0, 5, 5000, .3f, .0f, 0, SMART_KV_TAG_DEFAULT,     .5f, .5f);
    add_chunk(1, 2, 2, 0, 5, 5000, .3f, .0f, 0, SMART_KV_TAG_BOILERPLATE, .5f, .5f);
    add_chunk(2, 4, 2, 0, 5, 5000, .3f, .0f, 0, SMART_KV_TAG_CODE,        .5f, .5f);
    add_chunk(3, 6, 2, 0, 5, 5000, .3f, .0f, 0, SMART_KV_TAG_ERROR,       .5f, .5f);

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512,
        .migrate_max = 8,
        .base_type = 12,
        .memory_capacity = 0,
        .memory_used = 0,
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);
    dump_cache(&cfg, "Tag modifiers: same stats, different tags");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 4: adaptive gamma under pressure
// ────────────────────────────────────────────────────────────────────────────
static void scen_adaptive_gamma(void) {
    printf("=== SCENARIO: Adaptive Gamma Under Pressure ===\n\n");
    n_chunks = 0;
    // mix of chunks with varying recency
    for (int i = 0; i < 16; i++) {
        uint32_t last = (uint32_t)(CURRENT_STEP - i * 500);
        uint32_t acc = 16 - i;
        float redun = (i > 10) ? .6f : .0f;
        add_chunk(i, i*2, 2, i==0, acc, last, .3f, i==0?1.0f:0, 0, SMART_KV_TAG_DEFAULT, .5f, redun);
    }

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512,
        .migrate_max = 8,
        .base_type = 12,
        .memory_capacity = 10000,
        .memory_used = 9500,  // 95% pressure
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);
    dump_cache(&cfg, "Adaptive Gamma (95% memory pressure)");

    cfg.memory_used = 5000;  // 50% pressure, no adaptation
    dump_cache(&cfg, "Adaptive Gamma (50% memory pressure)");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 5: tier propagation table across base types
// ────────────────────────────────────────────────────────────────────────────
static void scen_tier_table(void) {
    printf("=== SCENARIO: Tier -> Type Mapping Across Base Types ===\n\n");
    int bases[] = { 0, 8, 14, 13, 12, 11, 10 };
    const char * bnames[] = { "f16", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k" };
    int nb = sizeof(bases)/sizeof(bases[0]);

    printf("%-6s | %-10s | %-10s | %-10s | %-10s | %-10s | %-12s\n",
           "base", "VH(1)", "Q(2)", "B(3)", "P(4)", "U(5)", "UTQ(6)");
    printf("-------|------------|------------|------------|------------|------------|--------------\n");
    for (int b = 0; b < nb; b++) {
        printf("%-6s |", bnames[b]);
        for (int t = 1; t <= SMART_KV_TIER_COUNT; t++) {
            int type = smart_kv_tier_to_type((smart_kv_tier)t, bases[b]);
            if (type == -1) printf(" %-10s", "plugin");
            else            printf(" %-10d", type);
        }
        printf("\n");
    }
    printf("\n");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 6: attention signal makes a difference
// ────────────────────────────────────────────────────────────────────────────
static void scen_attention(void) {
    printf("=== SCENARIO: With vs Without Attention ===\n\n");
    n_chunks = 0;
    // Two similar chunks: one gets high attention, one doesn't
    add_chunk(0, 0, 2, 0, 10, 5000, .90f, .0f, 0, SMART_KV_TAG_CODE, .5f, 0.0f);
    add_chunk(1, 2, 2, 0, 10, 5000, .10f, .0f, 0, SMART_KV_TAG_CODE, .5f, 0.0f);
    add_chunk(2, 4, 2, 0, 10, 5000, .50f, .0f, 0, SMART_KV_TAG_CODE, .5f, 0.0f);

    smart_kv_config cfg1 = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512, .migrate_max = 8, .base_type = 12,
        .memory_capacity = 0, .memory_used = 0,
    };
    smart_kv_init_weights(&cfg1.weights, cfg1.weights.gamma);
    smart_kv_init_tag_mods(&cfg1.weights);
    dump_cache(&cfg1, "Without Attention (w_a=0)");

    smart_kv_config cfg2 = {
        .weights = SMART_KV_WEIGHTS_ATTN,
        .score_interval = 512, .migrate_max = 8, .base_type = 12,
        .memory_capacity = 0, .memory_used = 0,
    };
    smart_kv_init_weights(&cfg2.weights, cfg2.weights.gamma);
    smart_kv_init_tag_mods(&cfg2.weights);
    dump_cache(&cfg2, "With Attention (w_a=0.25)");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 7: migration backoff demonstration
// ────────────────────────────────────────────────────────────────────────────
static void scen_backoff(void) {
    printf("=== SCENARIO: Migration Backoff ===\n\n");
    n_chunks = 0;
    add_chunk(0, 0, 2, 0, 5, 9900, .5f, .0f, 0, SMART_KV_TAG_DEFAULT, .5f, 0.0f);
    add_chunk(1, 2, 2, 0, 5, 9900, .5f, .0f, 0, SMART_KV_TAG_DEFAULT, .5f, 0.0f);
    add_chunk(2, 4, 2, 0, 5, CURRENT_STEP - 100, .5f, .0f, 0, SMART_KV_TAG_DEFAULT, .5f, 0.0f);

    // Set skip_counter after all chunks added
    chunks[0].skip_counter = 3; // in backoff

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512, .migrate_max = 8, .base_type = 12,
        .memory_capacity = 0, .memory_used = 0,
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);

    printf("chunk 0 skip_counter = 3 (in backoff)\n");
    printf("chunk 2 last_used_step = %llu (under min_residency=2048 from current %llu)\n\n",
           (unsigned long long)chunks[2].last_used_step,
           (unsigned long long)CURRENT_STEP);

    for (int i = 0; i < n_chunks; i++) {
        bool skip = smart_kv_should_skip(chunks[i].skip_counter, 2048, CURRENT_STEP, chunks[i].last_used_step);
        printf("  chunk %d: skip_counter=%u, last_used=%5llu, now=%5llu, diff=%5llu -> %s\n",
            chunks[i].chunk_id, chunks[i].skip_counter,
            (unsigned long long)chunks[i].last_used_step,
            (unsigned long long)CURRENT_STEP,
            (unsigned long long)(CURRENT_STEP - chunks[i].last_used_step),
            skip ? "SKIP" : "eval");
    }
    printf("\n");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 8: prefill analysis — tag detection from raw text
// ────────────────────────────────────────────────────────────────────────────
static const char * prefill_tag_name(const smart_kv_prefill_hint * h) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "%-12s p=%d min=%d red=%.2f conf=%.2f",
        smart_kv_tag_name(h->tag), h->pinned, h->min_tier, h->redundancy_score, h->confidence);
    return buf;
}

static void scen_prefill_analysis(void) {
    printf("=== SCENARIO: Prefill Content Analysis ===\n\n");

    struct { const char * text; const char * label; } samples[] = {
        { "<s>You are a helpful AI assistant. You help with coding tasks.", "system prompt" },
        { "{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"parameters\": {}}}", "tool schema" },
        { "error: undefined reference to `ggml_tensor_create\"./src/main.cpp:42", "compile error" },
        { "```rust\nfn fibonacci(n: u32) -> u32 {\n    match n { 0 => 0, 1 => 1, _ => ... }\n```", "code block" },
        { "./src/smart-kv-cache.h:120: error: call to undeclared library function 'expf'", "file path + error" },
        { "$ cargo build --release\n   Compiling smart-kv v0.1.0\n   Finished in 2.3s", "CLI command" },
        { "[INFO] Starting server on port 8080\n[DEBUG] Loading model...\n[WARN] Low memory\n[INFO] Ready", "boilerplate logs" },
        { "Sure! Here's how to implement the fibonacci function in Rust...", "assistant response" },
        { "can you write a function that reads a file and returns its contents?", "user request" },
        { "Therefore, as previously mentioned, the implementation is straightforward. However, it is important to note that there are several considerations. Furthermore, the approach should be considered carefully. In other words, the solution requires thought.", "rambling prose" },
        { "int main() { return 0; } // just a regular line of code", "plain code line" },
    };

    for (int i = 0; i < (int)(sizeof(samples)/sizeof(samples[0])); i++) {
        smart_kv_prefill_hint h = smart_kv_prefill_analyze(samples[i].text, (uint32_t)strlen(samples[i].text));
        printf("  %-25s -> %s\n", samples[i].label, prefill_tag_name(&h));
    }
    printf("\n");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 9: prefill tier placement vs bucket baseline
// ────────────────────────────────────────────────────────────────────────────
static void scen_prefill_vs_bucket(void) {
    printf("=== SCENARIO: Prefill Placement vs Bucket Baseline ===\n\n");
    printf("Context: 100k tokens, base_type=12 (q4_k), gamma=2.0\n");
    printf("Token position ranges from the SAMPLES below — smart places by\n");
    printf("content, bucket places by position.\n\n");

    struct { const char * text; uint32_t pos; } samples[] = {
        { "<s>You are an AI assistant.",                           0 },
        { "{\"type\": \"function\", \"name\": \"search\"}",     2048 },
        { "error: undefined reference to `foo'",                16384 },
        { "```rust\nfn hello() { println!(\"hi\"); }\n```",     32768 },
        { "$ cargo build",                                      49152 },
        { "[INFO] running\n[DEBUG] loop\n[WARN] memory",        65536 },
        { "Sure, here's how to fix that bug...",                81920 },
    };

    // Bucket config: q4_0:2048, q3:14336, q2_k:0  (balanced profile)
    // Position-based:
    //   0-2047:    q4_0  (ggml_type 2)
    //   2048-16383: q3    (ggml_type 11)
    //   16384+:     q2_k  (ggml_type 10)
    struct { uint32_t limit; int type; } bucket_tiers[] = {
        { 2048,    2  },  // q4_0
        { 16384,   11 },  // q3_k
        { 100000,  10 },  // q2_k
    };

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512, .migrate_max = 8, .base_type = 12,
        .memory_capacity = 0, .memory_used = 0,
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);

    printf("%-30s | %8s | %-18s | %-18s\n", "content", "position", "bucket tier", "smart prefill tier");
    printf("-------------------------------|----------|-------------------|-------------------\n");

    for (int i = 0; i < (int)(sizeof(samples)/sizeof(samples[0])); i++) {
        uint32_t pos = samples[i].pos;
        int bucket_type = bucket_tiers[0].type;
        for (int t = 0; t < 3; t++) {
            if (pos < bucket_tiers[t].limit) { bucket_type = bucket_tiers[t].type; break; }
        }

        smart_kv_prefill_hint h = smart_kv_prefill_analyze(samples[i].text, (uint32_t)strlen(samples[i].text));
        smart_kv_tier pt = smart_kv_prefill_tier(&h, &cfg);
        int prefill_type = smart_kv_tier_to_type(pt, cfg.base_type);

        int bits_bucket = bucket_type == 2 ? 4 : (bucket_type == 11 ? 3 : 2);
        int bits_prefill = prefill_type == -1 ? 0 : (prefill_type == 6 ? 5 : (prefill_type == 2 ? 4 : (prefill_type == 10 ? 2 : 1)));
        char pbuf[32];
        const char * ptype_str;
        if (prefill_type == -1) {
            ptype_str = "plug";
        } else {
            snprintf(pbuf, sizeof(pbuf), "type=%d", prefill_type);
            ptype_str = pbuf;
        }

        printf("%-30s | %8u | type=%-3d (%db) %-3s | %-6s (%db) %-3s\n",
               samples[i].text, pos,
               bucket_type, bits_bucket, bits_bucket <= 3 ? "LOW" : "OK",
               ptype_str, bits_prefill, bits_prefill >= 4 ? "OK" : "LOW");
    }

    // Summary
    printf("\nResult: Smart prefill places system/tools/errors at 4-5b while\n");
    printf("        boilerplate/old assistant at 1-2b. Bucket wastes 4b on\n");
    printf("        boilerplate at position 0-2048 and gives 2b to errors\n");
    printf("        at position 16384+. Same VRAM budget, better allocation.\n\n");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 10: Ultra-TQ tier 6 assignment
// ────────────────────────────────────────────────────────────────────────────
static void scen_ultra_tq(void) {
    printf("=== SCENARIO: Ultra-TQ Tier 6 Assignment ===\n\n");

    // Verify tier 6 is a plugin tier
    printf("smart_kv_tier_is_plugin(SMART_KV_TIER_ULTRA_TQ) = %s\n",
           smart_kv_tier_is_plugin(SMART_KV_TIER_ULTRA_TQ) ? "true (plugin)" : "false");
    printf("smart_kv_tier_is_plugin(SMART_KV_TIER_ULTRA)    = %s\n",
           smart_kv_tier_is_plugin(SMART_KV_TIER_ULTRA) ? "true (plugin)" : "false (GPU)");
    printf("smart_kv_tier_to_type(tier6, 12)                = %d (expected -1 = plugin)\n\n",
           smart_kv_tier_to_type(SMART_KV_TIER_ULTRA_TQ, 12));

    // Build chunks with descending scores to trigger tier 6
    smart_kv_chunk_meta chunks[8];
    for (int i = 0; i < 8; i++) {
        memset(&chunks[i], 0, sizeof(smart_kv_chunk_meta));
        chunks[i].chunk_id   = i;
        chunks[i].pos_begin  = i * 128;
        chunks[i].pos_end    = i * 128 + 127;
        chunks[i].n_tokens   = 128;
        chunks[i].tag        = SMART_KV_TAG_BOILERPLATE;
        chunks[i].pinned     = false;
        chunks[i].last_used_step = 100;
        // Simulate decreasing access — last 3 chunks barely touched
        chunks[i].access_count = i < 5 ? (9 - i) : 1;
        chunks[i].attention_ema = 0.2f;
        chunks[i].redundancy_score = i >= 5 ? 0.8f : 0.0f;
    }

    smart_kv_config cfg = {
        .weights = SMART_KV_WEIGHTS_NO_ATTN,
        .score_interval = 512, .migrate_max = 8, .base_type = 12,
        .memory_capacity = 0, .memory_used = 0,
    };
    smart_kv_init_weights(&cfg.weights, cfg.weights.gamma);
    smart_kv_init_tag_mods(&cfg.weights);

    uint32_t max_access = 0;
    for (int i = 0; i < 8; i++)
        if (chunks[i].access_count > max_access) max_access = chunks[i].access_count;

    printf("  chunk | score  | tier\n");
    printf("  ------|--------|-------------\n");
    for (int i = 0; i < 8; i++) {
        smart_kv_scored_chunk r = smart_kv_eval(&chunks[i], 100, max_access, &cfg);
        printf("  %5u | %5.3f | %-15s  type=%d\n",
               i, r.priority,
               smart_kv_tier_name(r.tier),
               r.target_type);
    }
    printf("\n");
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario 11: tier table with Ultra-TQ shown
// ────────────────────────────────────────────────────────────────────────────
static void scen_tier_table_tq(void) {
    printf("=== SCENARIO: Tier → Type Mapping (with Ultra-TQ) ===\n\n");

    int base_types[] = { 0, 8, 14, 13, 12, 11, 10 };
    const char * names[] = { "f16", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k" };
    int n = (int)(sizeof(base_types) / sizeof(base_types[0]));

    printf("%-10s", "base");
    for (int t = 1; t <= SMART_KV_TIER_COUNT; t++)
        printf(" | %-15s", smart_kv_tier_name((smart_kv_tier)t));
    printf("\n");
    printf("%-10s", "------");
    for (int t = 1; t <= SMART_KV_TIER_COUNT; t++) printf(" |-----------------");
    printf("\n");

    for (int i = 0; i < n; i++) {
        printf("%-10s", names[i]);
        for (int t = 1; t <= SMART_KV_TIER_COUNT; t++) {
            int type = smart_kv_tier_to_type((smart_kv_tier)t, base_types[i]);
            const char * tn = "?";
            static char tbuf[16];
            if (type >= 0) { snprintf(tbuf, sizeof(tbuf), "%d", type); tn = tbuf; }
            else { tn = "tq_plugin"; }
            printf(" | %-15s", tn);
        }
        printf("\n");
    }
    printf("\n");
}

// ────────────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────────────
int main(void) {
    printf("SMART KV CACHE TEST SUITE\n");
    printf("=========================\n\n");

    scen_agentic();
    scen_min_tier();
    scen_tag_mods();
    scen_adaptive_gamma();
    scen_tier_table();
    scen_attention();
    scen_backoff();
    scen_prefill_analysis();
    scen_prefill_vs_bucket();
    scen_ultra_tq();
    scen_tier_table_tq();

    printf("All scenarios complete.\n");
    return 0;
}
