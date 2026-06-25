# Building smart-tq-plugin

## Quick build

```batch
build_plugin.bat
```

Output: `build\smart-tq-plugin.dll` (~236 KB)

## Prerequisites

- ROCm clang++ (`C:\Program Files\AMD\ROCm\7.1\bin\clang++.exe`)
  — the same compiler used for the main llama.cpp build
- Ninja (optional, for implicit `build` directory creation)
- No external DLL dependencies — turboquant.cpp is compiled directly in

## What builds

| Source | Role |
|--------|------|
| `smart-tq-plugin.cpp` | Plugin entry point + KV cache lifecycle |
| `smart-kv-cache.c` | Heuristic scorer + tier mapping |
| `../turbo-quant-plugin/turboquant.cpp` | TurboQuant CPU encode/decode (tier 6) |

Both learned-score.h and smart-kv-cache.h are header-only — no .c needed.

## Post-build

Copy to plugin root for `start.bat` to pick it up:

```batch
copy build\smart-tq-plugin.dll smart-tq-plugin.dll
```

Or set `LLAMA_PLUGIN_FILE=plugins.env` pointing to the build path.

## If it fails

- **"use of undeclared identifier"**: function ordering issue in learned-score.h
  — `qr_init_random` / `rd_init_random` must be defined before `ls_load_all_weights`
- **"unsupported option '-fPIC'"**: Windows target, remove `-fPIC`
- **"fopen is deprecated"**: add `#define _CRT_SECURE_NO_WARNINGS` before includes
