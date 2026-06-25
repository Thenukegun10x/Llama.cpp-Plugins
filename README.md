# Llama.cpp Plugins

Unofficial plugin system for [llama.cpp](https://github.com/ggerganov/llama.cpp) on Windows, providing loadable runtime DLLs that extend server functionality.

## Plugins

| Plugin | Description |
|--------|-------------|
| **sage-attention-plugin** | HIP attention plugin, activated with `--plugin-attn on` |
| **turbo-quant-plugin** | KV-cache compression plugin (not enabled by default) |
| **smart-kv-plugin** | Smart KV-cache management with TQ integration |

## How it works

Plugins are loaded at runtime via `plugins.env` or the `LLAMA_PLUGIN_FILE` environment variable on Windows. Each plugin lives in its own folder under `plugins/` and contains the compiled DLL alongside source files and documentation.

- `plugins.env` — specifies which DLLs to load
- `LLAMA_PLUGIN_FILE` — alternative env var pointing to a plugin list file
- Avoid `LLAMA_PLUGIN_PATH` on Windows due to drive-letter colons causing parse issues

## Usage

1. Clone this repo into your llama.cpp server directory.
2. Set `plugins.env` or `LLAMA_PLUGIN_FILE` to the desired plugin DLLs.
3. Start the server — plugins load automatically.

---

> **Unofficial.** Not affiliated with the llama.cpp project.
> Source: https://github.com/Thenukegun10x/Llama.cpp-Plugins
