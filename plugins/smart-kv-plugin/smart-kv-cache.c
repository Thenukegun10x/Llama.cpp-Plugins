#include "smart-kv-cache.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/*
 * Quality table: ordered best-to-worst.
 * Add new types here and they automatically participate in tier mapping.
 * Platform-specific types (FP8, NVFP4, MXFP4, TQ) should be added
 * at the appropriate quality position. The tier step offsets will
 * select them correctly if the base type is at the right index.
 *
 * FP8 (E4M3): ~0.992 quality, 8 bits. Position between q8_0 and q6_k.
 *   GPU-native on H100/AMD MI350+. No quant overhead when GPU stores
 *   directly in FP8. Add as: { 0, "fp8", 0.992f } when a GGML_TYPE
 *   enum exists or via plugin-registered type.
 *
 * NVFP4 (E4M3 scale, 4 blocks): ~0.94 quality, 4 bits.
 *   { 0, "nvfp4", 0.940f } after q4_0.
 *
 * TQ4/TQ3 (TurboQuant): ~0.92/0.90 quality.
 *   { 0, "tq4", 0.920f } after q3_k
 *   { 0, "tq3", 0.900f } after q2_k
 */
static const struct { int type; const char * name; float quality; } qtable[] = {
    { 0,  "f16",   1.000f },
    { 8,  "q8_0",  0.995f },
    // { ?,  "fp8",   0.992f },  // GPU FP8 (E4M3) — add when GGML type exists
    { 14, "q6_k",  0.990f },
    { 13, "q5_k",  0.980f },
    { 7,  "q5_1",  0.975f },
    { 6,  "q5_0",  0.970f },
    { 12, "q4_k",  0.960f },
    { 2,  "q4_0",  0.945f },
    // { ?,  "nvfp4", 0.940f },  // NVIDIA FP4 — add when GGML type exists
    { 11, "q3_k",  0.930f },
    // { ?,  "tq4",   0.920f },  // TurboQuant Q4 — add when GGML type exists
    { 10, "q2_k",  0.880f },
    // { ?,  "tq3",   0.900f },  // TurboQuant Q3 — add when GGML type exists
    { 22, "iq2_s", 0.840f },
    { 19, "iq1_s", 0.780f },
};
#define QTABLE_LEN (sizeof(qtable)/sizeof(qtable[0]))

static int find_qidx(int t) {
    for (int i = 0; i < (int)QTABLE_LEN; i++)
        if (qtable[i].type == t) return i;
    return -1;
}

// --- Default weight sets ---

const smart_kv_weights SMART_KV_WEIGHTS_NO_ATTN = {
    .w_recency    = 0.45f,
    .w_attention  = 0.00f,
    .w_frequency  = 0.20f,
    .w_query      = 0.10f,
    .w_pin        = 0.30f,
    .w_redundancy = 0.20f,
    .tau_r        = 8192.0f,
    .lambda_a     = 0.95f,
    .gamma        = 2.0f,
};

const smart_kv_weights SMART_KV_WEIGHTS_ATTN = {
    .w_recency    = 0.35f,
    .w_attention  = 0.25f,
    .w_frequency  = 0.15f,
    .w_query      = 0.10f,
    .w_pin        = 0.25f,
    .w_redundancy = 0.20f,
    .tau_r        = 8192.0f,
    .lambda_a     = 0.95f,
    .gamma        = 2.0f,
};

void smart_kv_init_weights(smart_kv_weights * w, float gamma) {
    float t[SMART_KV_TIER_COUNT] = { 0.80f, 0.55f, 0.32f, 0.18f, 0.08f };
    float inv_gamma = 1.0f / gamma;
    for (int i = 0; i < SMART_KV_TIER_COUNT; i++)
        w->priority_thresholds[i] = powf(t[i], inv_gamma);
    w->gamma = gamma;
    if (w->tag_mod_recency[0] == 0.0f)
        smart_kv_init_tag_mods(w);
}

void smart_kv_init_tag_mods(smart_kv_weights * w) {
    for (int i = 0; i < SMART_KV_MAX_TAGS; i++) {
        w->tag_mod_recency[i]    = 1.0f;
        w->tag_mod_frequency[i]  = 1.0f;
        w->tag_mod_pin[i]        = 1.0f;
        w->tag_mod_redundancy[i] = 1.0f;
    }
    // Built-in heuristic tag modifiers
    w->tag_mod_recency[SMART_KV_TAG_ERROR]       = 1.5f;  // errors stay fresh
    w->tag_mod_recency[SMART_KV_TAG_SYSTEM]      = 1.3f;
    w->tag_mod_recency[SMART_KV_TAG_FILE_PATH]   = 1.3f;
    w->tag_mod_recency[SMART_KV_TAG_COMMAND]     = 1.2f;
    w->tag_mod_frequency[SMART_KV_TAG_CODE]      = 1.5f;  // code gets extra freq weight
    w->tag_mod_frequency[SMART_KV_TAG_TOOL_SCHEMA] = 1.5f;
    w->tag_mod_pin[SMART_KV_TAG_SYSTEM]           = 1.5f;  // system prompt strongly pinned
    w->tag_mod_pin[SMART_KV_TAG_TOOL_SCHEMA]      = 1.3f;
    w->tag_mod_pin[SMART_KV_TAG_ERROR]            = 1.2f;
    w->tag_mod_redundancy[SMART_KV_TAG_BOILERPLATE] = 2.0f; // double redundancy penalty for boilerplate
    w->tag_mod_redundancy[SMART_KV_TAG_ASSISTANT]    = 1.5f; // assistant prose ages faster
}

// --- Prefill Analysis ---

smart_kv_prefill_hint smart_kv_prefill_analyze(const char * text, uint32_t len) {
    smart_kv_prefill_hint h;
    memset(&h, 0, sizeof(h));
    h.tag = SMART_KV_TAG_DEFAULT;
    h.min_tier = 0;
    h.confidence = 0.3f;
    h.anchor_score = 0.0f;

    if (!text || len == 0) return h;

    int strong_matches = 0;

    // ── Tier 1: high-priority pinned content ──

    // System prompt
    if (strstr(text, "<s>") || strstr(text, "<|im_start|>") ||
        strstr(text, "You are a") || strstr(text, "you are an AI")) {
        h.tag = SMART_KV_TAG_SYSTEM;
        h.pinned = true;
        h.min_tier = 1;
        h.anchor_score = 1.0f;
        strong_matches += 2;
        h.confidence = fminf(0.3f + 0.2f * (float)strong_matches, 0.95f);
        return h; // unambiguous
    }

    // Tool/function schemas (JSON-heavy, key patterns)
    if ((strstr(text, "\"type\": \"function\"") || strstr(text, "\"function\"")) &&
        (strstr(text, "\"parameters\"") || strstr(text, "\"description\""))) {
        h.tag = SMART_KV_TAG_TOOL_SCHEMA;
        h.pinned = true;
        h.min_tier = 1;
        h.anchor_score = 1.0f;
        strong_matches += 2;
        h.confidence = fminf(0.3f + 0.2f * (float)strong_matches, 0.95f);
        return h; // unambiguous
    }

    // Error messages
    if (strstr(text, "error:") || strstr(text, "Error:") ||
        strstr(text, "failed") || strstr(text, "Failed") ||
        strstr(text, "traceback") || strstr(text, "Traceback") ||
        strstr(text, "exception") || strstr(text, "Exception") ||
        strstr(text, "exit code") || strstr(text, "non-zero") ||
        strstr(text, "undefined reference") || strstr(text, "not found") ||
        strstr(text, "fatal error")) {
        h.tag = SMART_KV_TAG_ERROR;
        h.pinned = true;
        h.min_tier = 2;
        h.anchor_score = 0.9f;
        strong_matches += 2;
        h.confidence = fminf(0.3f + 0.2f * (float)strong_matches, 0.95f);
        return h; // unambiguous
    }

    // ── Tier 2: boilerplate detection ──
    {
        int log_lines = 0;
        for (uint32_t i = 0; i < len && i < 500; i++) {
            if (text[i] == '\n' && i + 6 < len) {
                const char * rest = text + i + 1;
                if ((rest[0] == '[' && (strstr(rest, "INFO") || strstr(rest, "DEBUG") || strstr(rest, "WARN") || strstr(rest, "ERROR") || strstr(rest, "TRACE"))) ||
                    strstr(rest, "LOG ") ||
                    strstr(rest, " at "))
                    log_lines++;
            }
        }
        // Also check first line
        if (len > 4 && text[0] == '[') {
            if (strstr(text, "INFO") || strstr(text, "DEBUG") || strstr(text, "WARN") || strstr(text, "ERROR"))
                log_lines++;
        }
        if (log_lines >= 3) {
            h.tag = SMART_KV_TAG_BOILERPLATE;
            h.redundancy_score = fminf(0.5f + 0.1f * (float)log_lines, 1.0f);
            strong_matches++;
            h.confidence = fminf(0.3f + 0.2f * (float)strong_matches, 0.95f);
            return h;
        }
    }

    // ── Tier 3: user vs assistant ──

    // User input markers
    if (strstr(text, "can you") || strstr(text, "Can you") ||
        strstr(text, "write ") || strstr(text, "Write ") ||
        strstr(text, "create ") || strstr(text, "Create ") ||
        strstr(text, "how do ") || strstr(text, "How do ") ||
        strstr(text, "what is ") || strstr(text, "What is ") ||
        strstr(text, "explain ") || strstr(text, "Explain ") ||
        strstr(text, "implement") || strstr(text, "Implement") ||
        strstr(text, "fix ") || strstr(text, "Fix ") ||
        strstr(text, "add ") || strstr(text, "Add ") ||
        strstr(text, "change ") || strstr(text, "Change ") ||
        (strstr(text, "?") && len < 200)) {  // short text with question mark
        h.tag = SMART_KV_TAG_USER_INPUT;
        strong_matches++;
    }

    // Assistant response markers (check after user, higher tag number only wins if not user)
    if (h.tag == SMART_KV_TAG_DEFAULT) {
        if (strstr(text, "Here's") || strstr(text, "Here is") ||
            strstr(text, "I'll") || strstr(text, "I will") ||
            strstr(text, "Sure") || strstr(text, "Let me") ||
            strstr(text, "This is") || strstr(text, "We can") ||
            strstr(text, "I'd") || strstr(text, "I would") ||
            strstr(text, "You can") || strstr(text, "you can") ||
            strstr(text, "First,") || strstr(text, "first,") ||
            strstr(text, "Note:") || strstr(text, "note:")) {
            h.tag = SMART_KV_TAG_ASSISTANT;
            strong_matches++;
        }
    }

    // ── Tier 4: file paths, code, commands ──

    // File paths (only if not already tagged)
    if (h.tag == SMART_KV_TAG_DEFAULT || h.tag == SMART_KV_TAG_CODE) {
        if (strstr(text, ".\\") || strstr(text, "./") ||
            strstr(text, ":/") || strstr(text, ":\\") ||
            strstr(text, ".cpp") || strstr(text, ".h\"") ||
            strstr(text, ".py") || strstr(text, ".rs") ||
            strstr(text, ".ts") || strstr(text, ".js") ||
            strstr(text, ".c\"") || strstr(text, ".hpp") ||
            strstr(text, "src/") || strstr(text, "include/")) {
            h.tag = SMART_KV_TAG_FILE_PATH;
            h.anchor_score = fmaxf(h.anchor_score, 0.7f);
            h.min_tier = 3;
            strong_matches++;
        }
    }

    // Code blocks
    if (h.tag == SMART_KV_TAG_DEFAULT) {
        if (strstr(text, "```") ||
            strstr(text, "fn ") ||
            strstr(text, "struct ") ||
            strstr(text, "class ") ||
            strstr(text, "impl ") ||
            strstr(text, "pub ") ||
            strstr(text, "=>") ||
            strstr(text, " -> ")) {
            h.tag = SMART_KV_TAG_CODE;
            strong_matches++;
        }
    }

    // CLI commands
    if (h.tag == SMART_KV_TAG_DEFAULT) {
        if (strstr(text, "$ ") ||
            strstr(text, "> ") ||
            strstr(text, "npm ") ||
            strstr(text, "cargo ") ||
            strstr(text, "python ") ||
            strstr(text, "git ") ||
            strstr(text, "cmake ") ||
            strstr(text, "--flag") ||
            strstr(text, "--cache") ||
            strstr(text, "`cmd")) {
            h.tag = SMART_KV_TAG_COMMAND;
            h.anchor_score = fmaxf(h.anchor_score, 0.4f);
            strong_matches++;
        }
    }

    // ── Redundancy penalty for rambling prose ──
    if (h.tag == SMART_KV_TAG_ASSISTANT || h.tag == SMART_KV_TAG_USER_INPUT || h.tag == SMART_KV_TAG_DEFAULT) {
        int total_markers = 0;
        const char * markers[] = {
            "therefore", "however", "furthermore", "consequently",
            "importantly", "additionally", "notably", "specifically",
            "in other words", "that is", "in summary", "as mentioned",
            "for example", "in particular", "as a result"
        };
        for (int m = 0; m < (int)(sizeof(markers)/sizeof(markers[0])); m++) {
            const char * p = text;
            while ((p = strstr(p, markers[m])) != NULL) {
                total_markers++;
                p += strlen(markers[m]);
            }
        }
        if (total_markers > 5) {
            h.redundancy_score = fminf(h.redundancy_score + 0.3f, 1.0f);
            strong_matches++;
        }
    }

    h.confidence = fminf(0.3f + 0.2f * (float)strong_matches, 0.95f);
    return h;
}

// --- Core API ---

int smart_kv_tier_to_type(smart_kv_tier tier, int base_type) {
    if (tier == SMART_KV_TIER_ULTRA_TQ) return -1; // CPU offload via TQ plugin
    static const int steps[SMART_KV_TIER_COUNT] = { 0, 1, 3, 5, 7, 9 };
    int bi = find_qidx(base_type);

    // Unknown base type — find the closest entry by quality
    // (handles new GGML types added after compile time)
    if (bi < 0) {
        // Default: just use the mid-point as VH and step down from there
        int fallback = (int)QTABLE_LEN / 2;
        int ti = fallback + steps[tier - 1];
        if (ti >= (int)QTABLE_LEN) ti = (int)QTABLE_LEN - 1;
        return qtable[ti].type;
    }

    // Bound: don't go above F16 (index 0) for Very High
    if (tier == SMART_KV_TIER_VERY_HIGH && bi > 0) {
        return qtable[bi - 1].type; // one step above base
    }

    int ti = bi + steps[tier - 1];
    if (ti >= (int)QTABLE_LEN) ti = (int)QTABLE_LEN - 1;
    return qtable[ti].type;
}

const char * smart_kv_tier_name(smart_kv_tier t) {
    switch (t) {
        case SMART_KV_TIER_VERY_HIGH:   return "Very High";
        case SMART_KV_TIER_QUALITY:     return "Quality";
        case SMART_KV_TIER_BALANCED:    return "Balanced";
        case SMART_KV_TIER_PERFORMANCE: return "Performance";
        case SMART_KV_TIER_ULTRA:       return "Ultra";
        case SMART_KV_TIER_ULTRA_TQ:    return "Ultra-TQ (CPU)";
        default: return "?";
    }
}

const char * smart_kv_tag_name(uint8_t tag) {
    switch (tag) {
        case SMART_KV_TAG_DEFAULT:      return "default";
        case SMART_KV_TAG_SYSTEM:       return "system";
        case SMART_KV_TAG_TOOL_SCHEMA:  return "tool_schema";
        case SMART_KV_TAG_CODE:         return "code";
        case SMART_KV_TAG_ERROR:        return "error";
        case SMART_KV_TAG_USER_INPUT:   return "user_input";
        case SMART_KV_TAG_ASSISTANT:    return "assistant";
        case SMART_KV_TAG_BOILERPLATE:  return "boilerplate";
        case SMART_KV_TAG_FILE_PATH:    return "file_path";
        case SMART_KV_TAG_COMMAND:      return "command";
        default: return "custom";
    }
}

smart_kv_scored_chunk smart_kv_eval(const smart_kv_chunk_meta * c, uint64_t now,
    uint32_t max_access, const smart_kv_config * cfg)
{
    float p = smart_kv_score(c, now, max_access, &cfg->weights);
    float gamma = smart_kv_adaptive_gamma(cfg->weights.gamma, cfg->memory_used, cfg->memory_capacity);
    smart_kv_weights adj = cfg->weights;
    if (gamma != adj.gamma) {
        smart_kv_init_weights(&adj, gamma);
    }
    smart_kv_tier t = smart_kv_assign_tier(p, c->min_tier, adj.priority_thresholds);
    smart_kv_scored_chunk r = { .tier = t, .target_type = smart_kv_tier_to_type(t, cfg->base_type), .priority = p };
    return r;
}

void smart_kv_log(const smart_kv_chunk_meta * c, float score, smart_kv_tier t, int target) {
    const char * tn = "tq_plugin";
    if (target >= 0) {
        for (int i = 0; i < (int)QTABLE_LEN; i++)
            if (qtable[i].type == target) { tn = qtable[i].name; break; }
    }
    printf("  chunk %3u | pos %4u-%-4u | score %5.3f | tier %d (%-15s) | %-10s | min_tier=%d tag=%-12s pinned=%d access=%3u last=%5llu\n",
        c->chunk_id, c->pos_begin, c->pos_end, score,
        (int)t, smart_kv_tier_name(t), tn,
        c->min_tier, smart_kv_tag_name(c->tag),
        c->pinned, c->access_count, (unsigned long long)c->last_used_step);
}
