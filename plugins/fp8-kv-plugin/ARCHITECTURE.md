# fp8-kv-plugin — Architecture

## Purpose

Extend the smart-kv tiering system with FP8 as an intermediate GPU tier,
halving the per-slot VRAM cost for warm cache entries with zero quality loss.

## Tier Map

```
Tier 1-2: F16  (2 B/el, hot, lossless)
Tier 3-4: FP8  (1 B/el, warm, ~10% relative error)
Tier 5-6: TQ2  (0.43 B/el, cold, CPU offload)
```

The scorer (from smart-kv-plugin) assigns tiers based on attention frequency,
recency, and pressure. The base tensor type determines tier 1-4 cost; tier 5-6
is managed by the TQ plugin on CPU.

## FP8 is Opt-In (Default: F16 Fallback)

FP8 only activates when `FP8_KV_ENABLE=1` is set AND the hardware supports it
(gfx12+ / gfx942+, ROCm 6.2+). Without the env var, the plugin behaves
identically to smart-tq-plugin (F16 + TQ6 CPU offload).

## What the Plugin Does

1. **Detects FP8 tensor type** at init time (the ggml tensor must be F8_E4M3)
2. **Sets thresholds** in the scorer so tier 3-4 uses FP8 instead of F16
3. **Manages TQ6 CPU offload** (same as smart-tq-plugin)
4. **Reports stats**: F16 vs FP8 vs TQ6 slot counts per inference step

## What the Plugin Does NOT Do

- Does NOT allocate GPU memory (ggml handles the FP8 tensor)
- Does NOT convert F16↔F8 (ggml set_rows kernel handles conversion)
- Does NOT modify the tensor type (ggml core change required)

## Dependencies

| Component | Status |
|-----------|--------|
| ggml_type GGML_TYPE_F8_E4M3 | Phase 1 core patch |
| HIP F16↔F8 conversion kernels | Phase 1 core patch |
| F8 set_rows support | Phase 1 core patch |
| F8 attention kernel path | Phase 1 core patch |
| llama.cpp F8 cache type accept | Phase 1 core patch |
| This plugin | Phase 2 |

## File Map

```
fp8-kv-plugin/
├── fp8-kv-plugin.cpp     # Main plugin: init, scorer hooks, stats
├── fp8-kv-plugin.h       # (optional) shared types
├── build_plugin.bat      # Windows build script
├── ARCHITECTURE.md       # This file
├── BUILD.md              # Build instructions
├── PATCHES.md            # Full patch plan for ggml + llama.cpp
└── build/                # Build output directory
```
