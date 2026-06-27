# Building fp8-kv-plugin

## Prerequisites

- ROCm 6.2+ (for `__hip_fp8_e4m3` support)
- Visual Studio 2022 Build Tools
- llama.cpp built from source with `GGML_PLUGIN_SUPPORT=ON`

## Quick Start

```
build_plugin.bat
```

This compiles `fp8-kv-plugin.cpp` into `build/fp8-kv-plugin.dll`.

## Build Commands

### Debug
```
cl /nologo /O0 /Z7 /LD /I..\..\llama.cpp\ggml\include /DFP8_KV_PLUGIN_DEBUG fp8-kv-plugin.cpp /Fe:build\fp8-kv-plugin.dll
```

### Release
```
cl /nologo /O2 /LD /I..\..\llama.cpp\ggml\include fp8-kv-plugin.cpp /Fe:build\fp8-kv-plugin.dll
```

## Linking

This plugin has no runtime dependencies beyond the standard C runtime.
It uses the ggml plugin API from `ggml-plugin.h`.

## Runtime

Enable in `start.bat`:
```
set GGML_PLUGIN_LOAD=fp8-kv-plugin
set GGML_USE_PLUGIN_KV_CACHE=1
```

### FP8 mode (opt-in)
```
set FP8_KV_ENABLE=1
```

Without `FP8_KV_ENABLE=1`, the plugin runs in F16 fallback mode (same behavior
as smart-tq-plugin). This ensures older AMD GPUs (MI250, RDNA1-3) work
without changes.

### Requirements for FP8 mode
- `FP8_KV_ENABLE=1` environment variable
- ggml core patches (GGML_TYPE_F8_E4M3 + HIP kernels)
- ROCm >= 6.2
- gfx12+ (RDNA4) or gfx942+ (CDNA3) GPU

Without the core patches, the plugin logs a clear diagnostic on what's missing
and falls back to F16 gracefully.
