# Sage Attention Plugin

Runtime DLL:

```text
sage-attention-plugin.dll
```

Used by `start.bat` through `plugins.env` and `--plugin-attn on`.

Current behavior:

- Handles supported HIP Flash Attention shapes.
- Falls back to built-in Flash Attention if unsupported.
- Intended for RX 9070 XT / ROCm HIP builds.

Rebuild source:

```powershell
cmake --build ..\..\SageAttention-rocm\sage-attention-plugin\build --config Release -j 8
Copy-Item -LiteralPath "..\..\SageAttention-rocm\sage-attention-plugin\build\sage-attention-plugin.dll" -Destination ".\sage-attention-plugin.dll" -Force
```
