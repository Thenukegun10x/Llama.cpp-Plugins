# Llama.cpp Plugins

Unofficial plugin system for [llama.cpp](https://github.com/ggerganov/llama.cpp) on Windows/ROCm, providing loadable runtime DLLs that extend server functionality — tiered KV-cache eviction, FP8 tensor compression, flash attention, and more.

---

## Quick Start

```
start.bat fp8smart
```

Launches the server with FP8 KV cache + smart tiered eviction plugin. See `start.bat` for all presets.

---

## Plugins

### `smart-kv-plugin` (stratum-cache) — *loaded by default*

**DLL:** `plugins\smart-kv-plugin\smart-tq-plugin.dll`

A 6-tier scoring-based KV cache with self-learning MLP scorers:

| Tier | Storage | Location | Cost |
|------|---------|----------|------|
| 1-2 Very High / Quality | F16 (2 B/el) or FP8 (1 B/el) | GPU VRAM | Highest quality |
| 3-4 Balanced / Performance | FP8 (1 B/el) | GPU VRAM | 2x density vs F16 |
| 5 Ultra | FP8 (1 B/el) | GPU VRAM | More aggressive retention |
| 6 Ultra-TQ | TurboQuant (~0.43 B/el) | **CPU RAM** | VRAM-free |

Two MLP models trained from heuristic teacher labels:
- **QuantRegressor** (23→16→1, 401 params): assigns GPU tier 1-5
- **RAMDemoter** (24→8→1, 209 params): decides CPU offload (tier 6)

**Env vars:**

| Var | Default | Description |
|-----|---------|-------------|
| `SMART_KV_PROFILE` | `balanced` | Tier profile: `high`, `balanced`, `perf`, `ultra` |
| `SMART_KV_FP8` | `0` | Set `1` for FP8 GPU tensors (needs `--cache-type-k f8_e4m3`) |
| `SMART_KV_TRAIN_MODE` | `0` | Set `1` to collect training data (ring buffer → export) |
| `SMART_KV_SNAPSHOT_EXPORTS` | `0` | Periodic training data exports |
| `SMART_KV_EXPORT_DIR` | *(plugin dir)* | Export path for training data |

**Profiles** (`SMART_KV_PROFILE`):

| Profile | Offload Scale | RAM Threshold | Behavior |
|---------|--------------|---------------|----------|
| `high` / `very-high` | 0.50 | 0.95 | Max retention, minimal eviction |
| `balanced` | 1.00 | 0.85 | Default |
| `perf` / `performance` | 1.50 | 0.75 | More aggressive offload |
| `ultra` | 2.00 | 0.65 | Max VRAM efficiency |

---

### `fp8-kv-plugin` — *FP8 GPU tensor tier (opt-in)*

**DLL:** `plugins\fp8-kv-plugin\build\fp8-kv-plugin.dll`

Extends the smart-kv tier system with native `GGML_TYPE_F8_E4M3` GPU tensors. Requires AMD RDNA4 (RX 9070 XT) or CDNA3 GPU + ROCm 6.2+.

When FP8 is enabled, tiers become:
- **Tier 1-2:** F16 (2 B/el, hot/lossless)
- **Tier 3-4:** FP8 (1 B/el, ~10% relative error, 2x VRAM)
- **Tier 5-6:** TurboQuant (CPU offload)

Requires ggml core patches for `GGML_TYPE_F8_E4M3` type, HIP F16↔F8 conversion kernels, and F8 flash attention path. Applied to the llama.cpp source in this repo.

**Env var:** `FP8_KV_ENABLE=1` (opt-in)

---

### `sage-attention-plugin` — *HIP flash attention*

**DLL:** `plugins\sage-attention-plugin\sage-attention-plugin.dll`

ROCm HIP Flash Attention acceleration for RDNA4. Handles supported attention shapes natively; falls back to built-in flash attention for others.

**Activation:** `--plugin-attn on` (not loaded by default)

Built from: [SageAttention-rocm](https://github.com/Thenukegun10x/SageAttention-rocm)

---

### `turbo-quant-plugin` — *KV-cache compression engine*

**DLL:** `plugins\turbo-quant-plugin\build\Release\turbo-quant-plugin.dll`

PolarQuant (recursive polar-coordinate transform) + QJL (sign-bit residual correction) compression. Compresses F16 K/V vectors to ~5.44 bpw. Used as a library by `smart-kv-plugin` and `fp8-kv-plugin` for tier-6 CPU offload.

Not loaded directly — the compression code is compiled into the tiered cache plugins.

---

## Architecture

```
llama-server
 ├── built-in FLASH_ATTN (ggml-cuda)
 ├── --plugin-attn on → sage-attention-plugin (HIP flash attention)
 └── --plugin-kv-cache on → smart-kv-plugin (via plugins.env)
       ├── tiers 1-5: GPU F16/FP8
       │     └── fp8-kv-plugin opt-in: FP8 tiers (GGML_TYPE_F8_E4M3)
       └── tier 6: CPU TurboQuant (PolarQuant+QJL)
             └── turbo-quant-plugin compression engine
```

### Startup order

1. `start.bat` sets `HIP_VISIBLE_DEVICES=1`, `LLAMA_PLUGIN_FILE=plugins.env`
2. llama-server loads DLLs listed in `plugins.env`
3. Each plugin's `on_load()` fires → registers capabilities
4. `plugins.env` currently loads: `smart-tq-plugin.dll`

### Plugin loading

- `plugins.env` — specifies DLL paths and load order
- `LLAMA_PLUGIN_FILE` — alternative env var for the plugin list
- Plugins advertise capabilities via `GGML_PLUGIN_CAP_*`:
  - `GGML_PLUGIN_CAP_KV_CACHE` — intercepts KV cache store/retrieve
  - `GGML_PLUGIN_CAP_ATTN` — replaces flash attention compute

---

## Available presets (`start.bat`)

| Preset | KV Cache | Plugin | Profile | Context |
|--------|----------|--------|---------|---------|
| `default` / `fp8` | F8 | None | — | 32k |
| `f16` | F16 | None | — | 32k |
| `smart` | F16 | Smart-KV | balanced | 32k |
| `f16smart` | F16 | Smart-KV | ultra | 32k |
| `fp8smart` | F8 | Smart-KV | ultra | 32k |
| `fp8balanced` | F8 | Smart-KV | balanced | 32k |
| `fp8aggressive` | F8 | Smart-KV | perf | 32k |
| `fp8max` | F8 | Smart-KV | ultra | **128k** |
| `fp8train` | F8 | Smart-KV | balanced | 32k |
| `test` / `train` | F16 | Smart-KV | balanced | 32k |

---

## Benchmarks (RX 9070 XT, Qwen 3.5 4B Q8_0)

| Config | Prompt (pp512) | Gen (tg128, short ctx) | Gen (25k ctx) |
|--------|---------------|----------------------|--------------|
| F16 no plugin | 5817 t/s | 78.9 t/s | ~35 t/s |
| F8 no plugin | 5771 t/s | 72.1 t/s | ~33 t/s |
| F16 + Smart-KV | — | 69.4 t/s | ~33 t/s |
| F8 + Smart-KV | — | 61.9 t/s | ~33 t/s |

FP8 saves 50% KV cache VRAM with ~10% generation overhead. Smart-KV plugin adds ~11% overhead for tier scoring. At long context, attention O(n²) dominates.

---

## Building

Requires ROCm 7.1, CMake + Ninja, and Visual Studio 2022.

```
git clone --recurse-submodules https://github.com/Thenukegun10x/Llama.cpp-Plugins
cd llama.cpp
cmake -G Ninja .. -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAMDGPU_TARGETS=gfx1201
cmake --build . --target llama-server -j 8
```

Each plugin also has its own build script in its directory.

---

> **Unofficial.** Not affiliated with the llama.cpp project.
> Source: https://github.com/Thenukegun10x/Llama.cpp-Plugins
