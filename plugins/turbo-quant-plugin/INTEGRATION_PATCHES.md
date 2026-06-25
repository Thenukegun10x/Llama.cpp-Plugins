# Integration patches for TurboQuant KV cache plugin

Apply these patches to enable `--plugin-kv-cache on` in llama-server.
The plugin compresses K/V from F16 to TurboQuant format on write.

---

## Patch 1: `include/llama.h` — Add `use_plugin_kv_cache` param

### 1a. Add field to `llama_context_params`

After line 351 (`bool use_plugin_attn;`), insert:

```c
        bool   use_plugin_kv_cache;                     // enable plugin KV cache hooks
```

### 1b. Add default in `llama_context_default_params()`

After line 3482 (`/*.use_plugin_attn             =*/ false,`), insert:

```c
        /*.use_plugin_kv_cache          =*/ false,
```

---

## Patch 2: `src/llama-context.cpp` — Plugin lifecycle + KV params

### 2a. Add KV plugin lifecycle block after line 218

```cpp
    if (params.use_plugin_kv_cache) {
        ggml_plugin_scan();

        if (ggml_plugin_num_loaded() > 0) {
            ggml_plugin_model_info_t pinfo;
            pinfo.n_ctx_train = hparams.n_ctx_train;
            pinfo.n_head      = hparams.n_head();
            pinfo.n_head_kv   = hparams.n_head_kv();
            pinfo.head_dim    = hparams.n_embd_head_k();
            pinfo.n_layers    = hparams.n_layer();
            pinfo.n_embd      = hparams.n_embd;
            ggml_plugin_model_loaded(&pinfo);
        }
    }
```

### 2b. Override type_k/v to F16 when KV plugin is active

After line 351 (`/*.mem_other =*/ ...`), insert an override:

```cpp
        // if KV plugin is active, force F16 cache (plugin handles compression)
        if (params.use_plugin_kv_cache) {
            params_mem.type_k = GGML_TYPE_F16;
            params_mem.type_v = GGML_TYPE_F16;
        }
```

---

## Patch 3: `src/llama-kv-cache.h` — Add plugin fields

### 3a. Add to `kv_layer` struct (around line 230)

After `ggml_tensor * k` / `v` / stream views, add:

```cpp
    // plugin KV cache context (NULL if not using plugin)
    void * plugin_ctx_k;
    void * plugin_ctx_v;
```

Initialize both to `nullptr` in the constructor.

### 3b. Add to `llama_kv_cache` class (around line 80)

```cpp
    bool use_plugin;
```

Initialize to `false` in constructor initializer list.

### 3c. Add public method declarations

After the `get_v()` declaration (around line 280), add:

```cpp
    // plugin accessors
    bool  plugin_active() const { return use_plugin; }
    void *plugin_get_ctx_k(int32_t il) const;
    void *plugin_get_ctx_v(int32_t il) const;
```

---

## Patch 4: `src/llama-kv-cache.cpp` — Core hooks

### 4a. Constructor: init plugin layers (after tensor creation, around line 280)

```cpp
#ifdef GGML_PLUGIN_SUPPORT
    // initialize KV cache plugin layers
    use_plugin = false;
    {
        ggml_plugin_kv_cache_params_t kvp;
        kvp.n_embd_k_gqa = hparams.n_embd_k_gqa_all;
        kvp.n_embd_v_gqa = hparams.n_embd_v_gqa_all;
        kvp.kv_size      = kv_size;
        kvp.n_head_kv    = hparams.n_head_kv_all;
        kvp.head_dim_k   = hparams.n_embd_head_k();
        kvp.head_dim_v   = hparams.n_embd_head_v();
        kvp.device       = -1;

        for (auto & layer : layers) {
            kvp.layer_idx = (int)layer.il;
            layer.plugin_ctx_k = ggml_plugin_kv_cache_create_layer(
                kvp.layer_idx, &kvp, 4 /* bits_per_angle */);
            layer.plugin_ctx_v = ggml_plugin_kv_cache_create_layer(
                kvp.layer_idx, &kvp, 4 /* bits_per_angle */);

            if (layer.plugin_ctx_k || layer.plugin_ctx_v) {
                use_plugin = true;
            }
        }

        if (use_plugin) {
            fprintf(stderr, "[kv-plugin] KV cache plugin active for %zu layers\n",
                    layers.size());
        }
    }
#endif
```

### 4b. Destructor: destroy plugin layers (in `~llama_kv_cache`)

```cpp
#ifdef GGML_PLUGIN_SUPPORT
    for (auto & layer : layers) {
        if (layer.plugin_ctx_k) {
            ggml_plugin_kv_cache_destroy_layer(layer.plugin_ctx_k);
        }
        if (layer.plugin_ctx_v) {
            ggml_plugin_kv_cache_destroy_layer(layer.plugin_ctx_v);
        }
    }
#endif
```

### 4c. `clear()` : also clear plugin data

In `llama_kv_cache::clear()`, after `v_cells[s].clear()` (around line 402), add:

```cpp
#ifdef GGML_PLUGIN_SUPPORT
    if (use_plugin) {
        for (auto & layer : layers) {
            if (layer.plugin_ctx_k) {
                ggml_plugin_kv_cache_clear(layer.plugin_ctx_k);
            }
            if (layer.plugin_ctx_v) {
                ggml_plugin_kv_cache_clear(layer.plugin_ctx_v);
            }
        }
    }
#endif
```

### 4d. `apply_ubatch` — store to plugin after cell metadata update

In `apply_ubatch()`, after the cell metadata loop (after line 1134, before the function ends at line 1182), add:

```cpp
#ifdef GGML_PLUGIN_SUPPORT
    // if plugin is active, copy newly written K/V to plugin format
    if (use_plugin) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
                const uint32_t i     = s * sinfo.size() + ii;
                const auto idx = sinfo.idxs[s][ii];

                // for each layer, store K and V to plugin
                for (auto & layer : layers) {
                    if (layer.plugin_ctx_k) {
                        // k_cur data has been written to layer.k via cpy_k graph
                        // read the row from the cache tensor
                        const int64_t row_size = ggml_row_size(layer.k->type, layer.k->ne[0]);
                        const void * k_row = (const uint8_t*)layer.k->data + idx * row_size;
                        ggml_plugin_kv_cache_store(
                            layer.plugin_ctx_k, (int64_t)idx,
                            k_row, nullptr, 1);
                    }
                    if (layer.plugin_ctx_v) {
                        const int64_t row_size = ggml_row_size(layer.v->type, layer.v->ne[0]);
                        const void * v_row = (const uint8_t*)layer.v->data + idx * row_size;
                        ggml_plugin_kv_cache_store(
                            layer.plugin_ctx_v, (int64_t)idx,
                            nullptr, v_row, 1);
                    }
                }
            }
        }
    }
#endif
```

### 4e. Add helper methods

At the end of `llama-kv-cache.cpp` (before the closing namespace), add:

```cpp
#ifdef GGML_PLUGIN_SUPPORT
void * llama_kv_cache::plugin_get_ctx_k(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return nullptr;
    return layers[it->second].plugin_ctx_k;
}

void * llama_kv_cache::plugin_get_ctx_v(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return nullptr;
    return layers[it->second].plugin_ctx_v;
}
#endif
```

---

## Patch 5: `common/common.h` — Add field

After line 483:

```cpp
    bool   use_plugin_kv_cache   = false;     // enable plugin KV cache hooks
```

---

## Patch 6: `common/common.cpp` — Pass through

After line 1593:

```cpp
    cparams.use_plugin_kv_cache = params.use_plugin_kv_cache;
```

---

## Patch 7: `common/arg.cpp` — CLI arg `--plugin-kv-cache`

After the `--plugin-attn` block (after line 1410), add:

```cpp
    add_opt(common_arg(
        {"--plugin-kv-cache"}, "[on|off]",
        string_format("enable plugin KV cache hooks (default: '%s')",
                       is_truthy("false") ? "on" : "off"),
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.use_plugin_kv_cache = true;
            } else {
                params.use_plugin_kv_cache = false;
            }
        }
    ).set_env("LLAMA_ARG_PLUGIN_KV_CACHE"));
```

---

## Using it

In `start.bat`, replace the `--cache-type-k/v` lines:

```diff
--- a/start.bat
+++ b/start.bat
- --cache-type-k q4_0
- --cache-type-v q4_0
+ --plugin-kv-cache on
+ --cache-type-k f16
+ --cache-type-v f16

 set LLAMA_PLUGIN_PATH=.\plugins\turbo-quant-plugin.dll
```

The `--cache-type-k/v f16` ensures ggml uses F16 tensors (the plugin compresses independently). Set `LLAMA_PLUGIN_PATH` so the loader finds the DLL.

---

## How it works

1. At model init, `ggml_plugin_scan()` loads `turbo-quant-plugin.dll`
2. `llama_context_default_params()` now has `.use_plugin_kv_cache = false`
3. When `--plugin-kv-cache on`, the KV cache constructor creates per-layer plugin contexts
4. Tensors are created as F16 (type_k/v overridden)
5. After every batch writes K/V to the cache tensors (post-graph-eval), `apply_ubatch()` calls `ggml_plugin_kv_cache_store()` to compress F16 → TurboQuant
6. On cache clear, plugin data is cleared too
7. On unload, plugin contexts are freed

### Current limitation

The F16 tensor data is kept alongside the plugin's compressed data (2x memory for K/V). True memory savings require freeing the F16 tensor and decompressing on demand — that's a follow-up patch. This integration validates the pipeline.
