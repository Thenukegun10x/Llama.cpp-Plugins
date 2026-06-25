# Plugins

Plugins are stored one plugin per folder. Keep runtime DLLs beside a short README and, when useful, source/build files.

Current folders:

- `sage-attention-plugin`: HIP attention plugin used by `--plugin-attn on`.
- `turbo-quant-plugin`: KV-cache compression plugin source and DLL. It is not enabled by the launcher by default.

Use `plugins.env` or `LLAMA_PLUGIN_FILE` to load explicit DLLs on Windows. Avoid `LLAMA_PLUGIN_PATH` for absolute Windows paths because drive letters contain `:`.
