@echo off
setlocal enabledelayedexpansion

set HIP_VISIBLE_DEVICES=1
set LLAMA_PLUGIN_FILE=%~dp0plugins.env

:: Find newest .gguf in the server directory
set LATEST=
for /f "tokens=*" %%i in ('dir "%~dp0*.gguf" /b /o-d /a-d 2^>nul') do (
    if not defined LATEST set LATEST=%%i
)
if not defined LATEST (
    echo No .gguf model found in %~dp0
    pause
    exit /b 1
)
set MODEL=%~dp0%LATEST%
echo Using model: %LATEST%

set COMMON=^
 -ngl 99 -fa on ^
 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 ^
 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 ^
 --mirostat 0 -c 32768 -b 256 -ub 128 ^
 --parallel 1 --host 0.0.0.0 --port 12345

if /i "%1"=="" (
    echo.
    echo Available presets:
    echo   start default  - FP8 kv cache
    echo   start f16      - F16 kv cache
    echo   start smart    - Smart-KV plugin (balanced profile)
    echo   start f16smart - F16 + Smart-KV Ultra
    echo   start fp8smart - FP8 + Smart-KV Ultra
    echo   start test     - F16 + train mode (dump ring buffer)
    echo.
    set PRESET=default
) else (
    set PRESET=%1
)

if /i "!PRESET!"=="default" (
    echo [Preset: FP8 KV cache]
    "%~dp0llama.cpp\build\bin\llama-server.exe" -m "!MODEL!" ^
     --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup ^
     --plugin-kv-cache off --plugin-attn off %COMMON%
)

if /i "!PRESET!"=="f16" (
    echo [Preset: F16 KV cache]
    "%~dp0llama.cpp\build\bin\llama-server.exe" -m "!MODEL!" ^
     --plugin-kv-cache off --plugin-attn off %COMMON%
)

if /i "!PRESET!"=="smart" (
    echo [Preset: Smart-KV plugin (balanced)]
    set SMART_KV_FP8=0
    set SMART_KV_PROFILE=balanced
    "%~dp0llama.cpp\build\bin\llama-server.exe" -m "!MODEL!" ^
     --plugin-kv-cache on --plugin-attn off %COMMON%
)

if /i "!PRESET!"=="f16smart" (
    echo [Preset: F16 + Smart-KV Ultra]
    set SMART_KV_FP8=0
    set SMART_KV_PROFILE=ultra
    "%~dp0llama.cpp\build\bin\llama-server.exe" -m "!MODEL!" ^
     --plugin-kv-cache on --plugin-attn off %COMMON%
)

if /i "!PRESET!"=="fp8smart" (
    echo [Preset: FP8 + Smart-KV]
    set SMART_KV_FP8=1
    set SMART_KV_PROFILE=ultra
    set FP8_KV_ENABLE=1
    "%~dp0llama.cpp\build\bin\llama-server.exe" -m "!MODEL!" ^
     --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup ^
     --plugin-kv-cache on --plugin-attn off %COMMON%
)

if /i "!PRESET!"=="test" (
    echo [Preset: F16 + Training Mode (dump ring buffer)]
    set SMART_KV_TRAIN_MODE=1
    set SMART_KV_SNAPSHOT_EXPORTS=1
    set SMART_KV_PROFILE=balanced
    "%~dp0llama.cpp\build\bin\llama-server.exe" -m "!MODEL!" ^
     --plugin-kv-cache on --plugin-attn off %COMMON%
)
