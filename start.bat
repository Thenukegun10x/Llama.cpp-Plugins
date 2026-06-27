@echo off
setlocal enabledelayedexpansion

set HIP_VISIBLE_DEVICES=1
set LLAMA_PLUGIN_FILE=%~dp0plugins.env

for /f "tokens=*" %%i in ('dir "%~dp0*.gguf" /b /o-d /a-d 2^>nul') do if not defined LATEST set LATEST=%%i
if not defined LATEST echo No .gguf model found & pause & exit /b 1
set MODEL=%~dp0%LATEST%
echo Using model: %LATEST%

set COMMON=-ngl 99 -fa on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 32768 -b 256 -ub 128 --parallel 1 --host 0.0.0.0 --port 12345

if not "%1"=="" set PRESET=%1 & goto :RUN
echo.
echo Available presets:
echo   start default  - FP8 kv cache
echo   start f16      - F16 kv cache
echo   start smart    - Smart-KV (balanced)
echo   start f16smart - F16 + Smart-KV Ultra
echo   start fp8smart - FP8 + Smart-KV Ultra
echo   start test     - F16 + train mode
echo.
set PRESET=default

:RUN
if /i "!PRESET!"=="default" goto :DEFAULT
if /i "!PRESET!"=="f16" goto :F16
if /i "!PRESET!"=="smart" goto :SMART
if /i "!PRESET!"=="f16smart" goto :F16SMART
if /i "!PRESET!"=="fp8smart" goto :FP8SMART
if /i "!PRESET!"=="test" goto :TEST
echo Unknown preset: !PRESET!
exit /b 1

:DEFAULT
echo [Preset: FP8 KV cache]
set PRESET_ARGS=--cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup --plugin-kv-cache off --plugin-attn off
goto :LAUNCH

:F16
echo [Preset: F16 KV cache]
set PRESET_ARGS=--plugin-kv-cache off --plugin-attn off
goto :LAUNCH

:SMART
echo [Preset: Smart-KV plugin (balanced)]
set SMART_KV_FP8=0
set SMART_KV_PROFILE=balanced
set PRESET_ARGS=--plugin-kv-cache on --plugin-attn off
goto :LAUNCH

:F16SMART
echo [Preset: F16 + Smart-KV Ultra]
set SMART_KV_FP8=0
set SMART_KV_PROFILE=ultra
set PRESET_ARGS=--plugin-kv-cache on --plugin-attn off
goto :LAUNCH

:FP8SMART
echo [Preset: FP8 + Smart-KV]
set SMART_KV_FP8=1
set SMART_KV_PROFILE=ultra
set FP8_KV_ENABLE=1
set PRESET_ARGS=--cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup --plugin-kv-cache on --plugin-attn off
goto :LAUNCH

:TEST
echo [Preset: F16 + Training Mode]
set SMART_KV_TRAIN_MODE=1
set SMART_KV_SNAPSHOT_EXPORTS=1
set SMART_KV_PROFILE=balanced
set PRESET_ARGS=--plugin-kv-cache on --plugin-attn off
goto :LAUNCH

:LAUNCH
"%~dp0llama.cpp\build\bin\llama-server.exe" -m "!MODEL!" !PRESET_ARGS! %COMMON%
