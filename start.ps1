param(
    [string]$Preset = "default"
)

$ROOT = $PSScriptRoot
if (!$ROOT.EndsWith('\')) { $ROOT += '\' }
$SERVER = Join-Path $ROOT "llama.cpp\build\bin\llama-server.exe"
$env:HIP_VISIBLE_DEVICES = "1"
$env:LLAMA_PLUGIN_FILE = Join-Path $ROOT "plugins.env"

# ---- Pick newest GGUF in the same folder ----
$latest = @(Get-ChildItem -Path $ROOT -Filter *.gguf | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
if (-not $latest) {
    Write-Host "No .gguf model found in: $ROOT"
    pause
    exit 1
}
$model = Join-Path $ROOT $latest
Write-Host "Using model: $latest"

if (-not (Test-Path $SERVER)) {
    Write-Host "llama-server.exe not found: $SERVER"
    pause
    exit 1
}

# ---- Auto-detect model family from GGUF filename ----
$modelFname = [System.IO.Path]::GetFileNameWithoutExtension($latest)
$modelFname = $modelFname -replace '[\s\-_,.]',''
$modelFamily = ($modelFname -split '[\d]')[0]
if (-not $modelFamily) { $modelFamily = $modelFname }
Write-Host "Detected model family: $modelFamily"

# ---- Find matching mmproj file ----
$mmproj = $null
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($latest)
$candidates = @(Get-ChildItem -Path $ROOT | Where-Object { $_.Name -like "*$baseName*mmproj*.gguf" } | Select-Object -First 1)
if (-not $candidates) {
    $candidates = @(Get-ChildItem -Path $ROOT -Filter "mmproj*.gguf" | Select-Object -First 1)
}
if ($candidates) {
    $mmproj = $candidates.FullName
    Write-Host "Found multimodal projector: $mmproj"
}

# ---- Common server args (model-aware defaults) ----
$common = "-ngl 99 -fa on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 32768 -b 256 -ub 128 --parallel 1 --host 0.0.0.0 --port 12345"
$modelExtra = @()

switch -wildcard ($modelFamily) {
    "gemma*" {
        Write-Host "[Model tweak: Gemma family -- MTP samplers]"
        $common = "-ngl 99 -fa on --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 32768 -b 256 -ub 128 --parallel 1 --host 0.0.0.0 --port 12345"
        $modelExtra += "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--model-draft", "$ROOT\MTP\gemma-4-12B-it-MTP-Q8_0.gguf"
        if ($mmproj) { $modelExtra += "--mmproj", $mmproj }
    }
    "qwen*" {
        Write-Host "[Model tweak: Qwen family -- larger ctx & batch]"
        $common = "-ngl 99 -fa on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 131072 -b 2048 -ub 512 --parallel 1 --host 0.0.0.0 --port 12345"
    }
    "deepseek*" {
        Write-Host "[Model tweak: DeepSeek -- MLA bf16 cache]"
        $common = "-ngl 99 -fa on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 65536 -b 512 -ub 256 --parallel 1 --host 0.0.0.0 --port 12345"
        $modelExtra += "--cache-type-k", "bf16", "--cache-type-v", "bf16"
    }
    "llama*" {
        if ($modelFname -match 'llama4') {
            Write-Host "[Model tweak: Llama 4 -- larger ctx]"
            $common = "-ngl 99 -fa on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 131072 -b 1024 -ub 256 --parallel 1 --host 0.0.0.0 --port 12345"
        }
    }
    "granite*", "minicpm*" {
        if ($mmproj) { $modelExtra += "--mmproj", $mmproj }
    }
    "nomic*", "bert*" {
        $modelExtra += "--embedding", "--pooling", "cls"
    }
}

# ---- Normalize preset ----
$presetKey = $Preset.ToLower().Trim() -replace '[\s\-_/\\]',''

# ---- Build command args based on preset ----
$presetArgs = @()
$smartKvFp8 = $null
$smartKvProfile = $null
$fp8KvEnable = $null
$smartKvCompress = $null
$smartKvTrain = $null
$smartKvExport = $null

switch ($presetKey) {
    { $_ -in 'default','fp8' } {
        Write-Host "[Preset: FP8 KV cache]"
        $presetArgs = "--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--no-warmup", "--plugin-kv-cache", "off", "--plugin-attn", "off"
    }
    'f16' {
        Write-Host "[Preset: F16 KV cache]"
        $presetArgs = "--plugin-kv-cache", "off", "--plugin-attn", "off"
    }
    'smart' {
        Write-Host "[Preset: Smart-KV balanced]"
        $smartKvFp8 = "0"; $smartKvProfile = "balanced"
        $presetArgs = "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    { $_ -in 'f16smart','smartf16' } {
        Write-Host "[Preset: F16 + Smart-KV ultra]"
        $smartKvFp8 = "0"; $smartKvProfile = "ultra"
        $presetArgs = "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    { $_ -in 'fp8smart','f8smart','smartfp8' } {
        Write-Host "[Preset: FP8 + Smart-KV ultra]"
        $smartKvFp8 = "1"; $smartKvProfile = "ultra"; $fp8KvEnable = "1"
        $presetArgs = "--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--no-warmup", "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    { $_ -in 'fp8balanced','balancedfp8','fp8light' } {
        Write-Host "[Preset: FP8 + Smart-KV balanced]"
        $smartKvFp8 = "1"; $smartKvProfile = "balanced"; $fp8KvEnable = "1"
        $presetArgs = "--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--no-warmup", "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    'fp8max' {
        Write-Host "[Preset: FP8 + Smart-KV ultra, 128k context]"
        $common = "-ngl 99 -fa on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 131072 -b 256 -ub 128 --parallel 1 --host 0.0.0.0 --port 12345"
        $smartKvFp8 = "1"; $smartKvProfile = "ultra"; $fp8KvEnable = "1"
        $presetArgs = "--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--no-warmup", "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    { $_ -in 'fp8aggressive','aggressivefp8' } {
        Write-Host "[Preset: FP8 + Smart-KV performance (aggressive eviction)]"
        $smartKvFp8 = "1"; $smartKvProfile = "performance"; $fp8KvEnable = "1"
        $presetArgs = "--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--no-warmup", "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    { $_ -in 'fp8compress','compressfp8' } {
        Write-Host "[Preset: FP8 + Smart-KV ultra + context compression]"
        $smartKvFp8 = "1"; $smartKvProfile = "ultra"; $fp8KvEnable = "1"; $smartKvCompress = "1"
        $presetArgs = "--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--no-warmup", "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    { $_ -in 'fp8train','trainfp8' } {
        Write-Host "[Preset: FP8 + Smart-KV training mode]"
        $smartKvFp8 = "1"; $smartKvTrain = "1"; $smartKvExport = "1"; $smartKvProfile = "balanced"; $fp8KvEnable = "1"
        $presetArgs = "--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--no-warmup", "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    { $_ -in 'test','train' } {
        Write-Host "[Preset: F16 + Smart-KV training mode]"
        $smartKvFp8 = "0"; $smartKvTrain = "1"; $smartKvExport = "1"; $smartKvProfile = "balanced"
        $presetArgs = "--plugin-kv-cache", "on", "--plugin-attn", "off"
    }
    default {
        Write-Host ""
        Write-Host "Available presets:"
        Write-Host "  start.ps1 default       - FP8 KV cache (no plugin)"
        Write-Host "  start.ps1 f16           - F16 KV cache (no plugin)"
        Write-Host "  start.ps1 smart         - F16 + Smart-KV balanced"
        Write-Host "  start.ps1 f16smart      - F16 + Smart-KV ultra"
        Write-Host ""
        Write-Host "  FP8 + Smart-KV variants:"
        Write-Host "  start.ps1 fp8smart      - FP8 + Smart-KV ultra (max quality)"
        Write-Host "  start.ps1 fp8compress   - FP8 + Smart-KV ultra + context compression"
        Write-Host "  start.ps1 fp8balanced   - FP8 + Smart-KV balanced"
        Write-Host "  start.ps1 fp8light      - same as fp8balanced"
        Write-Host "  start.ps1 fp8aggressive - FP8 + Smart-KV (more tier 4 evictions)"
        Write-Host "  start.ps1 fp8max        - FP8 + Smart-KV ultra, 128k ctx"
        Write-Host "  start.ps1 fp8train      - FP8 + Smart-KV train mode"
        Write-Host ""
        Write-Host "  start.ps1 test          - F16 + Smart-KV training/export mode"
        Write-Host ""
        Write-Host "Unknown preset: $Preset"
        exit 1
    }
}

# ---- Set env vars for plugins ----
if ($smartKvFp8)     { $env:SMART_KV_FP8 = $smartKvFp8 }
if ($smartKvProfile)  { $env:SMART_KV_PROFILE = $smartKvProfile }
if ($fp8KvEnable)     { $env:FP8_KV_ENABLE = $fp8KvEnable }
if ($smartKvCompress) { $env:SMART_KV_COMPRESS = $smartKvCompress }
if ($smartKvTrain)    { $env:SMART_KV_TRAIN_MODE = $smartKvTrain }
if ($smartKvExport)   { $env:SMART_KV_SNAPSHOT_EXPORTS = $smartKvExport }

# ---- Launch ----
Write-Host ""
Write-Host "Server: $SERVER"
Write-Host "Preset: $Preset"
Write-Host "Model: $latest"
if ($modelExtra.Count -gt 0) { Write-Host "Model extra: $($modelExtra -join ' ')" }
Write-Host ""

# Build and splat argument array
$allArgs = @()
$allArgs += "-m", $model
$allArgs += $modelExtra
$allArgs += $presetArgs
$allArgs += -split $common

& $SERVER $allArgs
exit $LASTEXITCODE
