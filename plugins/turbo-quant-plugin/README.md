# TurboQuant Plugin

Runtime DLL:

```text
turbo-quant-plugin.dll
```

This plugin exposes the KV-cache plugin capability. It is intentionally not loaded by `start.bat` because TurboQuant KV retrieval is not part of the active mixed-cache path yet.

Use only for experiments with:

```text
--plugin-kv-cache on
```

Notes:

- TQ3/TQ4 may be interesting for cold KV cache regions later.
- Current production launcher uses built-in mixed KV types instead.
- Keep this plugin disabled unless testing its KV-store/retrieve path directly.
