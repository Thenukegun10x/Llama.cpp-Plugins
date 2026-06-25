#include "smart-kv-cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_CHUNKS      1024
#define CHUNK_SIZE      128
#define TOTAL_STEPS     200000
#define SCORE_INTERVAL  512
#define MIGRATE_MAX     8
#define PIN_FIRST_STEPS 4096
#define PIN_LAST_STEPS  8192

typedef struct {
    smart_kv_chunk_meta chunks[MAX_CHUNKS];
    int n_chunks;
    uint64_t step;
    uint32_t total_tokens;
    uint32_t total_migrations;
    uint32_t n_score_cycles;

    // tier usage tracking
    uint64_t tier_usage[SMART_KV_TIER_COUNT];
    uint64_t promotions;
    uint64_t demotions;
    uint64_t pinned_hits;

    smart_kv_config cfg;
} kv_sim;

static uint32_t sim_max_access(kv_sim * sim) {
    uint32_t m = 0;
    for (int i = 0; i < sim->n_chunks; i++)
        if (sim->chunks[i].access_count > m) m = sim->chunks[i].access_count;
    return m;
}

static int sim_find_chunk_at(kv_sim * sim, uint32_t pos) {
    for (int i = 0; i < sim->n_chunks; i++) {
        if (pos >= sim->chunks[i].pos_begin && pos <= sim->chunks[i].pos_end)
            return i;
    }
    return -1;
}

static void sim_add_chunk(kv_sim * sim, uint8_t tag, bool pinned, float anchor) {
    if (sim->n_chunks >= MAX_CHUNKS) return;
    int i = sim->n_chunks++;
    memset(&sim->chunks[i], 0, sizeof(smart_kv_chunk_meta));
    sim->chunks[i].chunk_id = i;
    sim->chunks[i].pos_begin = sim->total_tokens;
    sim->chunks[i].pos_end = sim->total_tokens + CHUNK_SIZE - 1;
    sim->chunks[i].n_tokens = CHUNK_SIZE;
    sim->chunks[i].tag = tag;
    sim->chunks[i].pinned = pinned;
    sim->chunks[i].anchor_score = anchor;
    sim->chunks[i].last_used_step = sim->step;
    sim->chunks[i].access_count = 1;
    sim->chunks[i].attention_ema = 0.5f;
    sim->chunks[i].query_score = 0.5f;
    sim->chunks[i].redundancy_score = 0.0f;
    if (tag == SMART_KV_TAG_SYSTEM) sim->chunks[i].min_tier = 1;
    else if (tag == SMART_KV_TAG_TOOL_SCHEMA) sim->chunks[i].min_tier = 1;
    else if (tag == SMART_KV_TAG_ERROR) sim->chunks[i].min_tier = 2;
    else if (tag == SMART_KV_TAG_FILE_PATH) sim->chunks[i].min_tier = 3;
    sim->total_tokens += CHUNK_SIZE;
}

static void sim_access_chunk(kv_sim * sim, int idx) {
    if (idx < 0 || idx >= sim->n_chunks) return;
    sim->chunks[idx].access_count++;
    sim->chunks[idx].last_used_step = sim->step;
    // simulate attention: recent chunks get higher attention
    float attn = (float)rand() / (float)RAND_MAX;
    float lambda_a = sim->cfg.weights.lambda_a;
    sim->chunks[idx].attention_ema = lambda_a * sim->chunks[idx].attention_ema + (1.0f - lambda_a) * attn;
}

static void sim_auto_tag(kv_sim * sim, uint32_t pos, const char * text) {
    int idx = sim_find_chunk_at(sim, pos);
    if (idx < 0) return;
    smart_kv_chunk_meta * c = &sim->chunks[idx];
    if (strstr(text, "system") || strstr(text, "<s>")) {
        c->tag = SMART_KV_TAG_SYSTEM; c->pinned = true; c->min_tier = 1;
    } else if (strstr(text, "tool") || strstr(text, "function")) {
        c->tag = SMART_KV_TAG_TOOL_SCHEMA; c->pinned = true; c->min_tier = 1;
    } else if (strstr(text, "error") || strstr(text, "failed")) {
        c->tag = SMART_KV_TAG_ERROR; c->pinned = true; c->min_tier = 2;
    } else if (strstr(text, ".cpp") || strstr(text, ".h") || strstr(text, "/")) {
        c->tag = SMART_KV_TAG_FILE_PATH;
    } else if (strstr(text, "```")) {
        c->tag = SMART_KV_TAG_CODE;
    } else if (strstr(text, "LOG") || strstr(text, "DEBUG")) {
        c->tag = SMART_KV_TAG_BOILERPLATE; c->redundancy_score = 0.7f;
    } else if (strstr(text, "$ ") || strstr(text, "cmd")) {
        c->tag = SMART_KV_TAG_COMMAND;
    } else if (strstr(text, "user:")) {
        c->tag = SMART_KV_TAG_USER_INPUT;
    } else if (strstr(text, "assistant:")) {
        c->tag = SMART_KV_TAG_ASSISTANT;
    } else {
        c->tag = SMART_KV_TAG_DEFAULT;
    }
}

static void sim_score_cycle(kv_sim * sim) {
    sim->n_score_cycles++;
    uint32_t max_access = sim_max_access(sim);
    uint32_t new_used = 0;
    for (int i = 0; i < sim->n_chunks; i++) {
        new_used += sim->chunks[i].n_tokens;
    }
    sim->cfg.memory_used = new_used;

    int promote_count = 0, demote_count = 0, skip_count = 0;

    for (int i = 0; i < sim->n_chunks; i++) {
        smart_kv_chunk_meta * c = &sim->chunks[i];
        if (smart_kv_should_skip(c->skip_counter, 2048, sim->step, c->last_used_step)) {
            skip_count++;
            if (c->skip_counter > 0) c->skip_counter--;
            continue;
        }

        smart_kv_scored_chunk r = smart_kv_eval(c, sim->step, max_access, &sim->cfg);
        uint8_t old_tier = c->tier;
        uint8_t new_tier = (uint8_t)r.tier;

        if (new_tier != old_tier && old_tier != 0) {
            if (new_tier < old_tier) { promote_count++; sim->promotions++; }
            else { demote_count++; sim->demotions++; }
            c->skip_counter = 4; // backoff after migration
        }
        if (c->pinned) sim->pinned_hits++;
        c->tier = new_tier;
        c->target_tier = new_tier;
    }

    // Log summary every 10 cycles
    if (sim->n_score_cycles % 10 == 0 || sim->n_score_cycles == 1) {
        uint32_t used = 0, pinned_count = 0;
        int tc[SMART_KV_TIER_COUNT] = {0};
        for (int i = 0; i < sim->n_chunks; i++) {
            used += sim->chunks[i].n_tokens;
            if (sim->chunks[i].pinned) pinned_count++;
            uint8_t t = sim->chunks[i].tier;
            if (t >= 1 && t <= SMART_KV_TIER_COUNT) tc[t-1]++;
        }

        printf("[step %6llu | tokens %6u | chunks %3d] ", 
               (unsigned long long)sim->step, sim->total_tokens, sim->n_chunks);
        printf("tiers:");
        for (int t = 0; t < SMART_KV_TIER_COUNT; t++)
            printf(" %s=%d", smart_kv_tier_name((smart_kv_tier)(t+1)), tc[t]);
        printf(" | pinned=%d used=%u skip=%d promote=%d demote=%d\n",
               pinned_count, used, skip_count, promote_count, demote_count);
    }
}

static void sim_init(kv_sim * sim, int base_type, uint32_t capacity) {
    memset(sim, 0, sizeof(*sim));
    sim->step = 0;
    sim->total_tokens = 0;
    sim->cfg.weights = SMART_KV_WEIGHTS_NO_ATTN;
    smart_kv_init_weights(&sim->cfg.weights, sim->cfg.weights.gamma);
    smart_kv_init_tag_mods(&sim->cfg.weights);
    sim->cfg.score_interval = SCORE_INTERVAL;
    sim->cfg.migrate_max = MIGRATE_MAX;
    sim->cfg.base_type = base_type;
    sim->cfg.memory_capacity = capacity;
    sim->cfg.memory_used = 0;

    // seed RNG
    srand(42);
}

static void sim_run(kv_sim * sim) {
    printf("\n===== STRESS TEST: %d steps, capacity=%u =====\n\n",
           TOTAL_STEPS, sim->cfg.memory_capacity);

    for (sim->step = 0; sim->step <= TOTAL_STEPS; sim->step++) {
        // (A) Add new chunk periodically
        if (sim->total_tokens < 150000 && sim->step % (CHUNK_SIZE * 3) == 0) {
            int dice = rand() % 100;
            bool pinned = sim->total_tokens < PIN_FIRST_STEPS ||
                          sim->total_tokens + PIN_LAST_STEPS > TOTAL_STEPS * CHUNK_SIZE;
            uint8_t tag;
            float anchor = 0.0f;

            if (sim->total_tokens < 1024) {
                tag = SMART_KV_TAG_SYSTEM; pinned = true; anchor = 1.0f;
            } else if (dice < 5) {
                tag = SMART_KV_TAG_TOOL_SCHEMA; pinned = true; anchor = 1.0f;
            } else if (dice < 12) {
                tag = SMART_KV_TAG_ERROR; pinned = true; anchor = 0.9f;
            } else if (dice < 20) {
                tag = SMART_KV_TAG_FILE_PATH; anchor = 0.7f;
            } else if (dice < 35) {
                tag = SMART_KV_TAG_CODE;
            } else if (dice < 45) {
                tag = SMART_KV_TAG_COMMAND;
            } else if (dice < 60) {
                tag = SMART_KV_TAG_USER_INPUT;
            } else if (dice < 75) {
                tag = SMART_KV_TAG_ASSISTANT;
            } else if (dice < 85) {
                tag = SMART_KV_TAG_BOILERPLATE;
            } else {
                tag = SMART_KV_TAG_DEFAULT;
            }

            sim_add_chunk(sim, tag, pinned, anchor);
        }

        // (B) Access recent chunks (recency simulation)
        int n_accesses = (rand() % 5) + 1;
        for (int a = 0; a < n_accesses; a++) {
            if (sim->n_chunks == 0) break;
            // bias: 70% last 20 chunks, 30% random
            int idx;
            if (rand() % 100 < 70) {
                idx = sim->n_chunks - 1 - (rand() % (sim->n_chunks < 20 ? sim->n_chunks : 20));
            } else {
                idx = rand() % sim->n_chunks;
            }
            if (idx >= 0 && idx < sim->n_chunks)
                sim_access_chunk(sim, idx);
        }

        // (C) Periodic scoring
        if (sim->step > 0 && sim->step % SCORE_INTERVAL == 0) {
            sim_score_cycle(sim);
        }
    }

    // Final score
    sim_score_cycle(sim);
    printf("\n===== FINAL STATS =====\n");
    printf("Total steps:     %llu\n", (unsigned long long)sim->step);
    printf("Total chunks:    %d\n", sim->n_chunks);
    printf("Total tokens:    %u\n", sim->total_tokens);
    printf("Score cycles:    %u\n", sim->n_score_cycles);
    printf("Promotions:      %llu\n", (unsigned long long)sim->promotions);
    printf("Demotions:       %llu\n", (unsigned long long)sim->demotions);
    printf("Pinned hits:     %llu\n", (unsigned long long)sim->pinned_hits);

    int tc[SMART_KV_TIER_COUNT] = {0};
    uint32_t used = 0;
    for (int i = 0; i < sim->n_chunks; i++) {
        used += sim->chunks[i].n_tokens;
        uint8_t t = sim->chunks[i].tier;
        if (t >= 1 && t <= SMART_KV_TIER_COUNT) tc[t-1]++;
    }
    printf("Memory used:     %u / %u\n", used, sim->cfg.memory_capacity);
    printf("Tier distribution:\n");
    for (int t = 0; t < SMART_KV_TIER_COUNT; t++) {
        int type = smart_kv_tier_to_type((smart_kv_tier)(t+1), sim->cfg.base_type);
        printf("  %-15s (%d): %3d chunks -> ggml_type=%d\n",
               smart_kv_tier_name((smart_kv_tier)(t+1)), t+1, tc[t], type);
    }

    // Tag breakdown
    printf("\nTag breakdown:\n");
    int tag_counts[SMART_KV_MAX_TAGS] = {0};
    for (int i = 0; i < sim->n_chunks; i++) {
        uint8_t t = sim->chunks[i].tag;
        if (t < SMART_KV_MAX_TAGS) tag_counts[t]++;
    }
    for (int t = 0; t < SMART_KV_MAX_TAGS; t++) {
        if (tag_counts[t] > 0)
            printf("  %-15s: %d\n", smart_kv_tag_name((uint8_t)t), tag_counts[t]);
    }
}

int main(void) {
    printf("SMART KV CACHE STRESS TEST\n");
    printf("==========================\n");
    printf("Settings:\n");
    printf("  Total steps:      %d\n", TOTAL_STEPS);
    printf("  Chunk size:       %d\n", CHUNK_SIZE);
    printf("  Score interval:   %d\n", SCORE_INTERVAL);
    printf("  Migrate max:      %d\n", MIGRATE_MAX);
    printf("  Pin first:        %d steps\n", PIN_FIRST_STEPS);
    printf("  Pin last:         %d steps\n", PIN_LAST_STEPS);
    printf("  Max chunks:       %d\n\n", MAX_CHUNKS);

    // Test 1: no memory pressure
    kv_sim sim1;
    sim_init(&sim1, 12, 0);
    sim_run(&sim1);

    // Test 2: tight memory pressure
    kv_sim sim2;
    sim_init(&sim2, 12, 40000);
    sim_run(&sim2);

    printf("\nAll stress tests complete.\n");
    return 0;
}
