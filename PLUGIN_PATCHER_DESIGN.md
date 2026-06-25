# Plugin Auto-Patcher Framework

This document describes a source-level patch framework for installing llama.cpp
plugin integrations from the `plugins/` folder.

The goal is not to patch a compiled `llama-server.exe`. The patcher applies
normal Git patch files to a llama.cpp source checkout, then the user rebuilds
llama.cpp.

## Goals

- Let each plugin declare the host patches it requires.
- Keep shared host changes, such as the plugin ABI, de-duplicated.
- Use standard unified diff patch files instead of custom text rewriting.
- Let Git handle context matching, already-applied checks, and 3-way fallback.
- Produce clear errors when upstream llama.cpp changed too much.
- Keep runtime plugin selection separate from source patching.

## Non-Goals

- Runtime binary patching.
- Hooking arbitrary stock llama.cpp builds without source changes.
- Replacing Git's patch application logic with custom fuzzy matching.
- Automatically resolving semantic conflicts after upstream refactors.

## Directory Layout

Recommended layout:

```text
plugins/
  _shared-patches/
    plugin-abi/
      patch.yaml
      001-ggml-plugin-abi.patch
    mixed-kv-cache/
      patch.yaml
      001-mixed-kv-cache.patch

  sage-attention-plugin/
    sage-attention-plugin.dll
    plugin.patch.yaml
    patches/
      001-sage-attention-hook.patch

  turbo-quant-plugin/
    turbo-quant-plugin.dll
    plugin.patch.yaml
    patches/
      001-kv-cache-hook.patch

  smart-kv-plugin/
    smart-tq-plugin.dll
    plugin.patch.yaml
    patches/
      001-smart-tier-routing.patch

tools/
  auto_patch.py
```

Shared patches live under `plugins/_shared-patches/`. Plugin-specific patches
live inside the plugin folder.

## Manifest Format

Each plugin can include `plugin.patch.yaml`.

Example:

```yaml
schema: llama-plugin-patch/v1
plugin:
  id: smart-kv-plugin
  name: Smart KV Plugin
  version: 0.1

target:
  project: llama.cpp
  min_commit: null
  max_commit: null

requires:
  patches:
    - plugin-abi
    - kv-cache-hook
    - mixed-kv-cache

patches:
  - id: smart-tier-routing
    file: patches/001-smart-tier-routing.patch
    description: Route cold KV chunks to the Smart/TQ plugin bucket.
    depends_on:
      - plugin-abi
      - kv-cache-hook
      - mixed-kv-cache

runtime:
  dll: smart-tq-plugin.dll
  plugins_env: true
  flags:
    - --plugin-kv-cache on
```

Shared patch manifest example:

```yaml
schema: llama-plugin-patch/v1
patch:
  id: plugin-abi
  name: GGML plugin ABI
  version: 0.1
  files:
    - 001-ggml-plugin-abi.patch
  provides:
    - ggml-plugin-abi-v1
```

## Patch Identity

Every patch has a stable `id`.

Examples:

```text
plugin-abi
sage-attention-hook
kv-cache-hook
mixed-kv-cache
smart-tier-routing
```

Patch IDs are used to:

- de-duplicate shared requirements
- order dependency application
- report which patch failed
- track what has already been applied

The patcher should never apply the same patch ID twice in one run.

## Patch Files

Patch files should be standard unified diffs generated from the llama.cpp repo.

For modified tracked files:

```powershell
git diff -- path\to\file.cpp > patch.patch
```

For new files, mark them as intent-to-add before generating the diff:

```powershell
git add -N ggml\include\ggml-plugin.h ggml\src\ggml-plugin.c
git diff -- ggml\include\ggml-plugin.h ggml\src\ggml-plugin.c > patch.patch
```

The patcher should not rely on prose instructions in `PATCHES.md` files. Those
are useful for humans, but the patcher should apply real `.patch` files.

## Patcher Commands

Suggested CLI:

```powershell
python tools\auto_patch.py list
python tools\auto_patch.py plan --plugin smart-kv-plugin --target .\llama.cpp
python tools\auto_patch.py check --plugin smart-kv-plugin --target .\llama.cpp
python tools\auto_patch.py apply --plugin smart-kv-plugin --target .\llama.cpp
python tools\auto_patch.py env --plugin smart-kv-plugin
```

Optional convenience command:

```powershell
python tools\auto_patch.py install --plugin smart-kv-plugin --target .\llama.cpp --write-env
```

## Apply Flow

The Python patcher should do this:

```text
1. Locate repo root and target llama.cpp checkout.
2. Verify the target is a Git worktree.
3. Scan plugin manifests and shared patch manifests.
4. Build the dependency graph for selected plugins.
5. De-duplicate patches by patch ID.
6. Topologically sort patches.
7. For each patch:
   a. Check if already applied with git apply --reverse --check.
   b. If already applied, mark as skipped.
   c. Otherwise run git apply --check.
   d. If clean, run git apply.
   e. If not clean, try git apply --3way.
   f. If still failing, stop and print the failed patch ID and file.
8. If requested, write plugins.env entries for selected plugins.
9. Print rebuild instructions.
```

The patcher should stop on the first failed patch. Continuing after a conflict
can leave the source tree in a confusing partial state.

## Already-Applied Detection

For each patch file:

```powershell
git apply --reverse --check path\to\patch.patch
```

If that succeeds, the patch is already applied.

If it fails, test whether the patch can be applied:

```powershell
git apply --check path\to\patch.patch
```

If that succeeds, apply it:

```powershell
git apply path\to\patch.patch
```

If normal application fails, try a 3-way merge:

```powershell
git apply --3way path\to\patch.patch
```

## State Tracking

Git is the primary source of truth. A separate state file is optional.

If used, write it outside the llama.cpp source tree or under a clearly named
local metadata path:

```text
.plugin-patcher-state.json
```

Example:

```json
{
  "target": "C:/path/to/llama.cpp",
  "applied": [
    {
      "id": "plugin-abi",
      "patch": "plugins/_shared-patches/plugin-abi/001-ggml-plugin-abi.patch",
      "status": "applied"
    }
  ]
}
```

State files are for reporting only. They should not override Git checks.

## Runtime Environment

Source patching and runtime plugin loading are separate.

After source patches are applied and llama.cpp is rebuilt, runtime loading uses
`plugins.env` or `LLAMA_PLUGIN_FILE`.

On Windows, prefer `plugins.env` because absolute paths contain drive-letter
colons. Colon splitting makes `LLAMA_PLUGIN_PATH` fragile.

Example `plugins.env`:

```text
.\plugins\sage-attention-plugin\sage-attention-plugin.dll
.\plugins\smart-kv-plugin\smart-tq-plugin.dll
```

The patcher can write this file when called with `--write-env`.

## Conflict Handling

When a patch fails, report:

- plugin ID
- patch ID
- patch file path
- command that failed
- Git output
- whether `--3way` was attempted

Do not attempt C++ AST rewriting or broad regex fixes automatically. If a patch
fails because upstream changed the host code, the patch should be updated by a
human and regenerated.

## Safety Rules

- Do not run `git reset`, `git checkout --`, or other destructive commands.
- Do not modify unrelated files.
- Do not apply plugin patches to a dirty file unless the user allows it.
- Do not silently overwrite `plugins.env`; either merge entries or require
  `--force-env`.
- Always support a `check` or `plan` mode before `apply`.

## Patch Categories

Suggested shared patch IDs:

```text
plugin-abi
  Adds ggml-plugin.h, ggml-plugin.c, loader CMake entries, and capability API.

sage-attention-hook
  Adds Flash Attention dispatch to the plugin ABI.

kv-cache-hook
  Adds --plugin-kv-cache and llama_kv_cache store/clear/destroy hooks.

mixed-kv-cache
  Adds mixed precision KV cache source files and CLI options.

smart-tier-routing
  Adds Smart/TQ tier routing on top of mixed KV cache.
```

Plugin requirements can then be simple:

```text
sage-attention-plugin -> plugin-abi + sage-attention-hook
turbo-quant-plugin    -> plugin-abi + kv-cache-hook
smart-kv-plugin       -> plugin-abi + kv-cache-hook + mixed-kv-cache + smart-tier-routing
```

## Implementation Notes

Python should orchestrate only. Git should apply patches.

Recommended Python modules:

```text
argparse
dataclasses
json
pathlib
subprocess
sys
yaml or json
```

If avoiding external dependencies, use JSON manifests instead of YAML. YAML is
nicer for humans, but JSON avoids requiring PyYAML.

## Minimal JSON Manifest Alternative

```json
{
  "schema": "llama-plugin-patch/v1",
  "plugin": {
    "id": "sage-attention-plugin",
    "name": "Sage Attention Plugin",
    "version": "0.1"
  },
  "requires": {
    "patches": ["plugin-abi"]
  },
  "patches": [
    {
      "id": "sage-attention-hook",
      "file": "patches/001-sage-attention-hook.patch",
      "depends_on": ["plugin-abi"]
    }
  ],
  "runtime": {
    "dll": "sage-attention-plugin.dll",
    "plugins_env": true,
    "flags": ["--plugin-attn on"]
  }
}
```

## Recommended First Version

Start with a small patcher that supports:

- JSON manifests only.
- `list`, `plan`, `check`, and `apply` commands.
- `git apply --reverse --check` for already-applied detection.
- `git apply --check`, then `git apply`, then optional `git apply --3way`.
- `plugins.env` generation behind `--write-env`.

Add YAML, uninstall, rebuild integration, and version range checks later.
