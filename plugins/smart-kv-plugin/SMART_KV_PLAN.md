# Smart Weighted KV Cache Plan

This document describes a future smart KV cache policy for llama.cpp. It is intended to sit beside the current chronological mixed-bucket cache, not replace it.

The current mixed cache is simple:

```text
position range -> fixed compression bucket
```

The proposed smart cache is score driven:

```text
chunk importance score -> compression tier or migration target
```

The safest implementation is chunk-based, not token-based. Use 64, 128, or 256 token chunks. Per-token scoring is more precise but too expensive and causes too much migration overhead.

## Goals

- Keep important context at higher KV precision.
- Compress old low-value context below q4 when memory pressure is high.
- Preserve system prompt, active instructions, tool schemas, errors, file paths, and current task context.
- Allow non-linear compression decisions instead of fixed chronological buckets.
- Keep the current bucket cache as `bucket` mode.
- Add a new `smart` mode that uses scoring plus migration.
- Add a plugin capability for custom scoring policy, not custom storage.

## Non-Goals

- Do not start with per-token migration.
- Do not require attention weights from every Flash Attention call in the first version.
- Do not make TurboQuant the first storage backend.
- Do not replace existing q4/q5/q8 cache paths.
- Do not make quality depend on a plugin being present.

## Recommended Modes

```text
--cache-policy bucket
```

Current behavior. Chronological mixed cache using `--cache-type-k-mixed`.

```text
--cache-policy smart
```

Built-in weighted policy. Uses chunk metadata and built-in scoring.

```text
--cache-policy smart-plugin
```

Core cache still owns storage and migration. Plugin only computes scores or tier hints.

## Storage Model

Keep physical storage as tiers. Each tier can use a normal cache type.

Example:

```text
tier 0 hot:    q8_0
tier 1 warm:   q5_1
tier 2 cool:   q4_0
tier 3 cold:   q3_k
tier 4 frozen: q2_k or future TQ4/TQ3
```

Logical positions remain stable. Physical chunks can move between tiers.

```text
logical pos/chunk id -> physical tier + slot range
```

This needs an indirection table. The attention path asks for logical K/V; the cache resolves chunks from one or more tiers and merges/dequantizes as needed.

## Chunk Metadata

Each chunk should track:

```c
struct smart_kv_chunk_meta {
    uint32_t chunk_id;
    uint32_t pos_begin;
    uint32_t pos_end;
    uint32_t n_tokens;

    uint8_t  tier;
    uint8_t  target_tier;
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
    float attention_ema;
};
```

## Trusted Scoring Components

The policy should combine established cache ideas:

- LRU style recency with exponential decay.
- LFU/TinyLFU style frequency using logarithmic counts.
- Heavy-hitter attention scoring similar to H2O-style KV eviction.
- Attention sink protection similar to StreamingLLM.
- Query/context similarity using cosine similarity or BM25-style sparse matching.
- Pinning for system/developer/tool-schema anchors.

These are practical, known signals. They are not magic. Start with recency, pinning, and frequency; add attention and query scores after instrumentation exists.

## Formula

For chunk `i` at decode step `t`:

```text
S_i(t) = clamp01(
    w_r * R_i(t)
  + w_a * A_i(t)
  + w_f * F_i(t)
  + w_q * Q_i(t)
  + w_p * P_i
  - w_d * D_i(t)
)
```

Where:

```text
R_i(t) = exp(-(t - last_used_i) / tau_r)
```

This is standard exponential recency decay. It behaves like a smooth LRU.

```text
F_i(t) = log(1 + access_count_i) / log(1 + max_access_count)
```

This is LFU-style frequency with log compression so one frequently used chunk does not dominate forever.

```text
A_i(t) = lambda_a * A_i(t - 1) + (1 - lambda_a) * attention_mass_i(t)
```

This is an EMA of attention received by the chunk. It approximates heavy-hitter KV retention.

```text
Q_i(t) = cosine(embed(query_window), embed(chunk_i))
```

If embeddings are too expensive, use sparse lexical matching:

```text
Q_i(t) = BM25_like(query_terms, chunk_terms)
```

For code, sparse matching over identifiers, paths, symbols, and error tokens may be more useful than dense embeddings.

```text
P_i = pin_or_anchor_score
```

Pinned chunks get a fixed boost. Some chunks should never be demoted below a minimum tier.

```text
D_i(t) = redundancy_or_low_information_score
```

Repeated logs, boilerplate, whitespace-heavy output, duplicate code, or old assistant prose get penalized.

## Initial Weights

Start conservative:

```text
w_r = 0.35
w_a = 0.25
w_f = 0.15
w_q = 0.10
w_p = 0.25
w_d = 0.20
tau_r = 8192 decode steps
lambda_a = 0.95
```

If attention scores are not available yet:

```text
w_r = 0.45
w_f = 0.20
w_q = 0.10
w_p = 0.30
w_d = 0.20
```

Normalize final score with `clamp01`.

## Non-Linear Tier Mapping

Use non-linear mapping so high-value chunks are strongly protected:

```text
priority_i = S_i(t)^gamma
gamma = 2.0
```

Tier assignment:

```text
priority >= 0.80 -> q8_0
priority >= 0.55 -> q5_1
priority >= 0.32 -> q4_0
priority >= 0.18 -> q3_k
else             -> q2_k or future cold plugin tier
```

Equivalent bit-budget version:

```text
bits_i = min_bits + (max_bits - min_bits) * S_i(t)^gamma
```

Then choose the closest available tier.

## Pinned Regions

Pinning is mandatory for agentic coding.

Always protect:

- System prompt.
- Developer instructions.
- Tool schemas.
- Current user task.
- Current active file paths.
- Compiler/test errors.
- Recent tool outputs.
- First N tokens, usually 2048 to 4096.
- Last N tokens, usually 4096 to 8192.

Pinned chunks should have a minimum tier:

```text
system/developer/tool schema -> q8_0 minimum
current task/errors/files    -> q5_1 minimum
recent window                -> q5_1 or q8_0 minimum
```

## Agentic Coding Heuristics

Boost chunks containing:

- File paths.
- Function names.
- Class names.
- Stack traces.
- Build errors.
- Test failures.
- CLI commands.
- Explicit TODOs.
- User constraints.
- API signatures.
- JSON schemas.
- Tool call results from active files.

Penalize chunks containing:

- Repeated assistant prose.
- Old successful logs.
- Duplicate code blocks.
- Large unchanged file dumps.
- Whitespace-heavy text.
- Repeated dependency output.
- Old chain-of-thought-like reasoning text.

## Migration Policy

Migration should happen periodically, not every token.

Recommended schedule:

```text
score every 256 or 512 generated tokens
migrate at most M chunks per cycle
M = 4 to 16 initially
```

Use hysteresis to avoid thrashing:

```text
promote if target_tier is 2 levels better for 2 cycles
demote if target_tier is 1 level worse for 3 cycles
```

Minimum residency:

```text
do not migrate same chunk twice within 2048 tokens
```

## Attention Score Instrumentation

Best signal is attention mass received by each chunk.

Problem:

```text
Flash Attention usually does not expose full attention weights cheaply.
```

Practical stages:

```text
stage 1: no attention signal, use recency/frequency/pinning/query
stage 2: sample attention on selected decode steps only
stage 3: add optional backend counters for attention mass per chunk
stage 4: plugin attention kernels can optionally report chunk attention mass
```

Do not block the first smart cache on attention telemetry.

## Built-In CLI Proposal

Keep existing bucket mode:

```text
--cache-policy bucket
--cache-type-k-mixed q8_0:4096,q5_1:12288,q4_0:24576,q3_k:0
```

Add smart mode:

```text
--cache-policy smart
--cache-smart-tiers q8_0,q5_1,q4_0,q3_k,q2_k
--cache-smart-budget 100000
--cache-smart-chunk 128
--cache-smart-pin-first 4096
--cache-smart-pin-last 8192
--cache-smart-score-interval 512
--cache-smart-migrate-max 8
```

Optional weights:

```text
--cache-smart-weights recency=0.35,attention=0.25,frequency=0.15,query=0.10,pin=0.25,redundancy=0.20
```

## Plugin Capability Proposal

The plugin should not own KV storage in the first version. It should only score chunks or return tier hints.

New capability:

```text
GGML_PLUGIN_CAP_KV_POLICY
```

Interface sketch:

```c
typedef struct {
    uint32_t chunk_id;
    uint32_t pos_begin;
    uint32_t pos_end;
    uint32_t n_tokens;

    uint8_t current_tier;
    uint8_t min_tier;
    bool pinned;

    float recency_score;
    float attention_score;
    float frequency_score;
    float query_score;
    float anchor_score;
    float redundancy_score;
} ggml_plugin_kv_policy_chunk_t;

typedef struct {
    uint32_t n_chunks;
    uint32_t n_tiers;
    uint32_t current_step;
    const ggml_plugin_kv_policy_chunk_t * chunks;

    float * scores_out;
    uint8_t * target_tiers_out;
} ggml_plugin_kv_policy_params_t;

typedef struct {
    int version;
    void (*on_load)(void);
    void (*on_unload)(void);
    bool (*policy_supported)(const ggml_plugin_kv_policy_params_t * params);
    int  (*score_chunks)(const ggml_plugin_kv_policy_params_t * params);
} ggml_plugin_kv_policy_v1_t;
```

Core responsibilities:

- Maintain chunk metadata.
- Own all KV storage tiers.
- Enforce pinned minimum tiers.
- Enforce memory budgets.
- Perform migration.
- Fall back to built-in scoring if plugin fails.

Plugin responsibilities:

- Compute scores.
- Optionally suggest target tiers.
- Never touch raw KV tensors.
- Never block correctness.

This keeps plugins safe and easy to test.

## Implementation Plan

### Phase 1: Built-In Smart Metadata

- Add `llama_kv_cache_smart` beside `llama_kv_cache_mixed`.
- Reuse mixed cache buckets as physical tiers.
- Add chunk table and logical-to-physical map.
- Implement pin-first and pin-last.
- Implement recency, frequency, and anchor scores.
- No attention telemetry yet.

### Phase 2: Chunk Migration

- Add migration graph ops or CPU fallback for dequant/requant chunk movement.
- Migrate only at scoring intervals.
- Limit migrations per cycle.
- Add hysteresis and minimum residency.
- Add logs for promotions/demotions.

### Phase 3: Query and Redundancy Signals

- Tokenize chunks into lightweight terms.
- Track identifiers, paths, and error-like terms.
- Add sparse query similarity against the current prompt/window.
- Add duplicate/low-information penalties.

### Phase 4: Attention-Derived Heavy Hitters

- Add optional sampled attention telemetry.
- Update per-chunk attention EMA.
- Protect attention sinks and heavy hitters.
- Keep this optional so Flash Attention remains fast.

### Phase 5: KV Policy Plugin

- Add `GGML_PLUGIN_CAP_KV_POLICY`.
- Add `--plugin-kv-policy on`.
- Call plugin at scoring intervals.
- Validate plugin outputs.
- Fall back to built-in policy on failure.

### Phase 6: Cold-Tier Experiments

- Test `q3_k` and `q2_k` cold tail.
- Test TurboQuant only after live read/decode path is properly integrated.
- Prefer TQ4 before TQ3.

## Recommended First Smart Profile

For a 100k coding agent context:

```text
chunk size: 128
pin first: 4096
pin last: 8192
tiers: q8_0, q5_1, q4_0, q3_k, q2_k
score interval: 512
max migrations per interval: 8
```

Built-in score weights without attention:

```text
recency=0.45,frequency=0.20,query=0.10,pin=0.30,redundancy=0.20
```

Built-in score weights with attention:

```text
recency=0.35,attention=0.25,frequency=0.15,query=0.10,pin=0.25,redundancy=0.20
```

## Risks

- Migration may cost more than memory savings if too frequent.
- Bad scoring can compress critical instructions.
- Attention telemetry can slow down fast attention.
- Mixed quant tiers require robust dequant/requant paths.
- q2/q3 tails may hurt long-context exactness.

Mitigations:

- Pin critical chunks.
- Use conservative defaults.
- Add detailed logs.
- Add a dry-run scoring mode.
- Keep bucket mode available.
- Make plugin scoring optional.

## Debugging Output

Add logs like:

```text
smart_kv: chunk 12 score=0.91 tier=q8_0 pinned=system
smart_kv: chunk 48 score=0.37 migrate q5_1 -> q4_0 reason=old_low_attention
smart_kv: chunk 103 score=0.12 migrate q4_0 -> q2_k reason=redundant_log
```

Add stats:

```text
smart_kv: tiers q8_0=6144 q5_1=14336 q4_0=32768 q3_k=28672 q2_k=18080
smart_kv: migrated promote=2 demote=6 skipped_pinned=4
```

## Bottom Line

The best design is not a formula-only plugin that directly controls storage. The best design is:

```text
core storage + core migration + optional scoring plugin
```

This keeps correctness in llama.cpp while allowing experimentation with better weighting formulas.
