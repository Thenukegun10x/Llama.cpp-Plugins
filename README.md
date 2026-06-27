# Llama.cpp Plugins

Unofficial plugin system for [llama.cpp](https://github.com/ggerganov/llama.cpp) on Windows/ROCm, providing loadable runtime DLLs that extend server functionality — tiered KV-cache eviction, FP8 tensor compression, flash attention, and more.

> **⚠ Work in progress.** All plugins are experimental — expect rough edges, incomplete features, and breaking changes. Not production-ready.

---

## Quick Start

```
start.bat fp8smart
```

Launches the server with FP8 KV cache + smart tiered eviction plugin. See `start.bat` for all presets.

---

## Plugins

### `smart-kv-plugin` (stratum-cache) — *enabled by Smart-KV presets*

**DLL:** `plugins\smart-kv-plugin\build\smart-tq-plugin.dll`

A 6-tier scoring-based KV cache with self-learning MLP scorers. Tiers 1-5 stay in the selected GPU KV tensor (`f16` for F16 presets, `f8_e4m3` for FP8 presets). Tier 6 offloads cold slots to CPU RAM with TurboQuant:

| Tier | Decision | Storage | Location |
|------|----------|---------|----------|
| 1-2 Very High / Quality | Hottest/highest-value slots | Selected GPU KV type (`f16` or `f8_e4m3`) | GPU VRAM |
| 3-5 Balanced / Performance / Ultra | Lower-priority slots kept on GPU | Selected GPU KV type (`f16` or `f8_e4m3`) | GPU VRAM |
| 6 Ultra-TQ | Cold/offloaded slots | TurboQuant (~0.43 B/el) | CPU RAM |

Two MLP models trained from heuristic teacher labels:
- **QuantRegressor** (23→16→1, 401 params): assigns GPU tier 1-5
- **RAMDemoter** (24→8→1, 209 params): decides CPU offload (tier 6)

**Env vars:**

| Var | Default | Description |
|-----|---------|-------------|
| `SMART_KV_PROFILE` | `balanced` | Tier profile: `high`, `balanced`, `perf`, `ultra` |
| `SMART_KV_FP8` | `0` | Set `1` for FP8 GPU tensors (needs `--cache-type-k f8_e4m3`) |
| `SMART_KV_COMPRESS` | `0` | Set `1` to enable context compression (see below) |
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

### Context Compression (`SMART_KV_COMPRESS=1`)

When the KV cache fills past a tunable threshold, the plugin frees the
coldest contiguous block of slots so they can be reused, while preserving
recent context and the first slots used as attention sinks. The configured
`n_ctx` does not change.

**Benefits:**
- Prevents OOM at long context by freeing cold slots before pressure hits 100%
- Designed to minimize quality impact: only slots with low attention_ema + low access count + old last_used_step are dropped
- Transparent to harnesses — they still see `n_ctx` unchanged
- Works alongside all other features (FP8, TQ, tiers)

**When to use:**
- Long-running agent loops (hundreds of turns)
- Contexts approaching 80%+ KV fill
- Combined with `fp8compress` preset for max context density

**Tuning knobs** (env vars):

| Var | Default | Description |
|-----|---------|-------------|
| `SMART_KV_COMPRESS` | `0` | Enable context compression |
| `SMART_KV_COMPRESS_PRESSURE` | `0.80` | Start compressing at this fill fraction |
| `SMART_KV_COMPRESS_BLOCK` | `4` | Slots to free per compression event |
| `SMART_KV_COMPRESS_SINKS` | `4` | First N slots preserved as attention sinks |
| `SMART_KV_COMPRESS_INTERVAL` | `256` | Check compression every N tier decisions |
| `SMART_KV_COMPRESS_THRESHOLD` | `0.70` | Coldness threshold (0-1) — higher = fewer drops |

**Example — compress aggressively at 70% fill:**
```bat
set SMART_KV_COMPRESS=1
set SMART_KV_COMPRESS_PRESSURE=0.70
set SMART_KV_COMPRESS_BLOCK=16
set SMART_KV_COMPRESS_THRESHOLD=0.50
```

---

### `fp8-kv-plugin` — *experimental standalone FP8 KV plugin*

**DLL:** `plugins\fp8-kv-plugin\build\fp8-kv-plugin.dll`

Prototype standalone plugin for native `GGML_TYPE_F8_E4M3` GPU tensors. It is not loaded by `start.bat` or `plugins.env`; the active FP8 presets use llama.cpp `--cache-type-k f8_e4m3 --cache-type-v f8_e4m3` with `smart-tq-plugin`.

FP8 uses 1 byte per element vs F16's 2 bytes, so the KV tensor uses half the VRAM. Native FP8 requires ggml core patches, ROCm support, and AMD RDNA4 (RX 9070 XT) or CDNA3-class hardware.

Intended standalone tier plan:
- **Tier 1-2:** F16 (2 B/el, hot/lossless)
- **Tier 3-4:** FP8 (1 B/el, ~10% relative error, **half the VRAM of F16**)
- **Tier 5-6:** TurboQuant (CPU offload)

Requires ggml core patches for `GGML_TYPE_F8_E4M3` type, HIP F16↔F8 conversion kernels, and an F8 flash attention path.

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

PolarQuant (recursive polar-coordinate transform) + QJL (sign-bit residual correction) compression, inspired by Google's TurboQuant approach. Compresses F16 K/V vectors to ~5.44 bpw. Used as a library by `smart-kv-plugin` and `fp8-kv-plugin` for tier-6 CPU offload.

Not loaded directly — the compression code is compiled into the tiered cache plugins.

---

## Architecture

```
llama-server
 ├── built-in flash attention (-fa on)
 ├── --plugin-attn on → sage-attention-plugin (HIP flash attention)
 └── --plugin-kv-cache on → smart-tq-plugin (via plugins.env)
       ├── tiers 1-5: selected GPU KV tensor (F16 or F8_E4M3)
       └── tier 6: CPU TurboQuant (PolarQuant+QJL)
             └── turbo-quant code compiled into smart-tq-plugin
```

### Startup order

1. `start.bat` sets `HIP_VISIBLE_DEVICES=1`, `LLAMA_PLUGIN_FILE=plugins.env`
2. llama-server loads DLLs listed in `plugins.env`
3. Each plugin's `on_load()` fires → registers capabilities
4. `plugins.env` currently loads: `plugins\smart-kv-plugin\build\smart-tq-plugin.dll`

### Plugin loading

- `plugins.env` — specifies DLL paths and load order
- `LLAMA_PLUGIN_FILE` — alternative env var for the plugin list
- Plugins advertise capabilities via `GGML_PLUGIN_CAP_*`
- `GGML_PLUGIN_CAP_KV_CACHE` intercepts KV cache store/retrieve
- `GGML_PLUGIN_CAP_ATTN` replaces flash attention compute

---

## Available presets (`start.bat`)

| Preset | KV Cache | Plugin | Profile | Context | Compression |
|--------|----------|--------|---------|---------|-------------|
| `default` / `fp8` | F8 | None | — | 32k | No |
| `f16` | F16 | None | — | 32k | No |
| `smart` | F16 | Smart-KV | balanced | 32k | No |
| `f16smart` | F16 | Smart-KV | ultra | 32k | No |
| `fp8smart` | F8 | Smart-KV | ultra | 32k | No |
| `fp8compress` | F8 | Smart-KV | ultra | 32k | **Yes** |
| `fp8balanced` | F8 | Smart-KV | balanced | 32k | No |
| `fp8aggressive` | F8 | Smart-KV | perf | 32k | No |
| `fp8max` | F8 | Smart-KV | ultra | **128k** | No |
| `fp8train` | F8 | Smart-KV | balanced | 32k | No |
| `test` / `train` | F16 | Smart-KV | balanced | 32k | No |

---

## Benchmarks (RX 9070 XT, Qwen 3.5 4B Q8_0)

| Config | Prompt (pp512) | Gen (tg128, short ctx) | Gen (25k ctx) | VRAM (idle) | VRAM (25k ctx) |
|--------|---------------|----------------------|--------------|-------------|----------------|
| F16 no plugin | 5817 t/s | 78.9 t/s | ~33 t/s | — | — |
| F8 no plugin | 5771 t/s | 72.1 t/s | ~33 t/s | — | — |
| F16 + Smart-KV | — | 76.1 t/s | 32.4 t/s | 7155 MiB | 7313 MiB (+158 MiB) |
| F8 + Smart-KV | — | 61.9 t/s | 32.5 t/s | 6931 MiB | — |
| **F8 + Smart-KV + Compress** | — | — | — | **6764 MiB** | 6923 MiB (+159 MiB) |

**Long-context recall (25k tokens, 9 needles):** 100% for all configs tested.

FP8 uses **half the KV cache VRAM** of F16 (~10% generation overhead). Smart-KV plugin adds ~11% overhead for tier scoring. At long context (25k+), attention O(n²) dominates and all cache types converge to ~33 t/s.

**Compression benefit:** F8 + Compress starts with **6764 MiB** vs 7155 MiB (F16) — a **391 MiB** VRAM saving from the pre-allocated KV tensor alone. Compression drops cold slots transparently when fill exceeds 80%, keeping recent context and sink anchors intact.

---

## System requirements

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 7800X3D (8 cores, 4.2 GHz) |
| GPU | AMD Radeon RX 9070 XT (RDNA4, gfx1201, 16 GB VRAM) |
| RAM | 32 GB |
| OS | Windows 11 Enterprise, Build 26200 |
| ROCm | 7.1 |
| Build | CMake + Ninja, Visual Studio 2022 |

---

## Building

Requires ROCm 7.1, CMake + Ninja, and Visual Studio 2022.

```bat
git clone --recurse-submodules https://github.com/Thenukegun10x/Llama.cpp-Plugins.git
cd Llama.cpp-Plugins
cmake -S llama.cpp -B llama.cpp\build -G Ninja -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAMDGPU_TARGETS=gfx1201
cmake --build llama.cpp\build --target llama-server -j 8
```

Each plugin also has its own build script in its directory. `start.bat` expects `llama.cpp\build\bin\llama-server.exe`; update `plugins.env` if the checked-in absolute DLL path does not match your clone location.

---

> **Unofficial.** Not affiliated with the llama.cpp project.
> Source: https://github.com/Thenukegun10x/Llama.cpp-Plugins
