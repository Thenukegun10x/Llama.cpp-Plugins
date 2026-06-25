# StratumCache — Smart Tiered KV Cache

Scoring-based KV cache with self-learning MLP for llama.cpp.
6 tiers including TurboQuant CPU offload.

## Files

| File | Purpose |
|------|---------|
| `smart-kv-cache.h` | API: chunk meta, weights, config, inline scorers |
| `smart-kv-cache.c` | Implementation: quality table, tier mapping, tag defaults |
| `learned-score.h` | MLP scorer: forward pass, ring buffer, dataset export |
| `smart-tq-plugin.cpp/.dll` | Combined plugin: scoring + TQ CPU offload + data collection |
| `train_learned_score.py` | PyTorch training script (ROCm/CUDA) |
| `collect_training_data.py` | Multi-trial data collector via V1 API |
| `test-smart-kv-cache.c` | Standalone test with 11 scenarios |
| `test-tq-integration.cpp` | TQ format roundtrip + full pipeline test |
| `PATCHES.md` | Mixed cache routing patches for llama.cpp |

## Architecture

```
Heuristic scorer (default):
  S = 0.45·R + 0.20·F + 0.10·Q + 0.30·P - 0.20·D
  R = exp(-delta/8192) recency, F = frequency, Q = query,
  P = pin, D = redundancy

Learned scorers (optional, replace S):
  QuantRegressor: 18→16→1 MLP (321 params, ~1.3 KB)
  RAMDemoter:     19→8→1 MLP  (169 params, ~0.7 KB)
  Inputs: recency_log, frequency_log, attention_ema, query_score,
          10× tag_onehot, redundancy_score, chunk_age_ratio, pinned, gamma
  RAMDemoter also gets: memory_pressure
  Trained on actual usage patterns from previous sessions.

Tier mapping:
  Score ≥ 0.894 → Very High  (Q5_0,      5 bpw)
  Score ≥ 0.742 → Quality    (Q4_0,      4 bpw)
  Score ≥ 0.566 → Balanced   (Q2_K,      2.56 bpw)
  Score ≥ 0.424 → Performance(IQ1_S,     1.56 bpw)
  Score ≥ 0.283 → Ultra      (IQ1_S,     1.56 bpw)
  Score <  0.283 → Ultra-TQ (CPU, TQ,   5.44 bpw)
```

## Usage

### Profiles in start.bat

| Profile | Command | Context | Cache |
|---------|---------|---------|-------|
| ram | `start.bat` | 32k | Q4_0 uniform |
| balanced | `start.bat balanced` | 100k | Q4_0→Q3→Q2 buckets |
| old | `start.bat old` | 100k | Q5→Q4→Q3→Q2 buckets |
| smart | `start.bat smart` | 200k | Smart tiers + TQ CPU |
| test | `start.bat test` | 16k | F16 (data collection) |

### Training the Learned Scorer

```
# 1. Collect data (use a 4B model, fits entirely in VRAM)
start.bat test
python collect_training_data.py 5

# 2. Train on your 9070 XT
python train_learned_score.py training_data_4000.bin

# 3. Use with production model
start.bat smart --train   # trains on exit after each session
```

The `--train` flag enables post-session retraining. Without it, the server
runs normally and session data is exported but not trained on.

### Model Sizes

| Model | Architecture | Params | Weights |
|-------|------------|--------|---------|
| QuantRegressor | 18→16→1 | 321 | ~1.3 KB |
| RAMDemoter | 19→8→1 | 169 | ~0.7 KB |
| Combined | — | ~490 | ~2.0 KB |

Forward pass for both models: ~1μs per chunk (negligible vs attention at milliseconds).

## Per-Chunk Controls

| Field | Type | Description |
|-------|------|-------------|
| `min_tier` | uint8_t | Never demote below this tier (1-6). Errors pinned to tier 1 can't slip to Q2_K. |
| `tag` | uint8_t | 10 tags: system, tool_schema, code, error, boilerplate, etc. |
| `skip_counter` | uint32_t | Migration backoff: skip N scoring cycles. Prevents thrash. |
| `anchor_score` | float | 0-1. Unpinned chunks with high anchor get partial pin boost. |
| `redundancy_score` | float | 0-1. Boilerplate, logs, rambling prose get penalized. |

## Tag Weight Modifiers

| Tag | pin_mul | redun_mul | min_tier | base_score |
|-----|---------|-----------|----------|------------|
| system | 1.5x | 0.5x | 1 | 0.350 → V.High |
| tool_schema | 1.5x | 0.5x | 1 | 0.350 → V.High |
| error | 1.3x | 0.5x | 2 | 0.290 → Quality |
| file_path | 1.0x | 1.0x | 3 | 0.100 → Balanced |
| code | 1.0x | 0.8x | 0 | 0.140 → depends |
| boilerplate | 0.8x | 2.0x | 0 | -0.160 → Ultra-TQ |

## VRAM Impact (Qwen 27B, 64 layers)

| Context | Balanced | Smart-TQ | Saved |
|---------|----------|----------|-------|
| 100k | 4000 MB | 2800 MB | 1.2 GB |
| 200k | 8000 MB | 4800 MB | 3.2 GB |

Tier 6 data lives on CPU at 5.44 bpw via TurboQuant. The GPU tensor
for tier 6 is 1 cell (~2 KB) — the old full-size tensor wasted ~400 MB.

## Yet to Implement

- **Adaptive online training**: the MLP currently trains from saved session
  data. Future: continuous online SGD during inference so the model adapts
  within a single session without needing a restart.
- **Attention-feedback labels**: current labels are tier-based proxies.
  Real labels from actual attention weights would give cleaner signal.
  Requires hooking into the softmax output during graph eval.
- **Per-layer MLP**: separate MLP per decoder layer. Early layers and deep
  layers have different attention patterns. Same architecture, different
  weights. Would improve accuracy ~5-10%.
- **Per-head tiering**: GQA means 8 heads. A chunk could be tier 1 for
  head 3 but tier 6 for heads 0-2,4-7. Requires per-head metadata.
- **Cross-session weight averaging**: average weights across sessions
  instead of fine-tuning from the last one. Reduces overfit to any single
  session's quirks.
- **Automatic model size selection**: choose nano/medium/large/XL based on
  available training data count.
- **FP8 native type**: add GGML_TYPE_F8_E4M3 as a cache tier for native
  8-bit floating point with hardware support on RX 9070 XT.
