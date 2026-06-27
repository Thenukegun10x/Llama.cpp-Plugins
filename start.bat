@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem llama.cpp server launcher
rem Usage:
rem   start.bat                 -> default FP8 KV cache
rem   start.bat default
rem   start.bat f16
rem   start.bat smart
rem   start.bat f16smart        -> F16 + Smart-KV ultra
rem   start.bat fp8smart        -> FP8 + Smart-KV ultra
rem   start.bat test            -> Smart-KV training/export mode
rem ============================================================

set "ROOT=%~dp0"
set "SERVER=%ROOT%llama.cpp\build\bin\llama-server.exe"

set "HIP_VISIBLE_DEVICES=1"
set "LLAMA_PLUGIN_FILE=%ROOT%plugins.env"

rem ---- Pick newest GGUF in the same folder as this .bat ----
set "LATEST="
for /f "delims=" %%i in ('dir "%ROOT%*.gguf" /b /o-d /a-d 2^>nul') do (
    if not defined LATEST set "LATEST=%%i"
)

if not defined LATEST (
    echo No .gguf model found in:
    echo %ROOT%
    pause
    exit /b 1
)

set "MODEL=%ROOT%%LATEST%"
echo Using model: %LATEST%

if not exist "%SERVER%" (
    echo llama-server.exe not found:
    echo %SERVER%
    pause
    exit /b 1
)

rem ---- Common server args ----
set "COMMON=-ngl 99 -fa on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 32768 -b 256 -ub 128 --parallel 1 --host 0.0.0.0 --port 12345"

rem ---- Get preset. %~1 removes surrounding quotes if present. ----
set "PRESET=%~1"
if not defined PRESET set "PRESET=default"

rem ---- Normalize preset aliases: f16-smart, f16_smart, F16SMART all become f16smart ----
set "PRESET=!PRESET: =!"
set "PRESET=!PRESET:-=!"
set "PRESET=!PRESET:_=!"
set "PRESET=!PRESET:/=!"
set "PRESET=!PRESET:\=!"

if /i "!PRESET!"=="default"     goto :DEFAULT
if /i "!PRESET!"=="fp8"         goto :DEFAULT
if /i "!PRESET!"=="f16"         goto :F16
if /i "!PRESET!"=="smart"       goto :SMART
if /i "!PRESET!"=="f16smart"    goto :F16SMART
if /i "!PRESET!"=="smartf16"    goto :F16SMART
if /i "!PRESET!"=="fp8smart"    goto :FP8SMART
if /i "!PRESET!"=="f8smart"     goto :FP8SMART
if /i "!PRESET!"=="smartfp8"    goto :FP8SMART
if /i "!PRESET!"=="fp8balanced" goto :FP8BALANCED
if /i "!PRESET!"=="balancedfp8" goto :FP8BALANCED
if /i "!PRESET!"=="fp8light"    goto :FP8BALANCED
if /i "!PRESET!"=="fp8max"      goto :FP8MAX
if /i "!PRESET!"=="maxfp8"      goto :FP8MAX
if /i "!PRESET!"=="fp8aggressive" goto :FP8AGGRESSIVE
if /i "!PRESET!"=="aggressivefp8" goto :FP8AGGRESSIVE
if /i "!PRESET!"=="fp8train"    goto :FP8TRAIN
if /i "!PRESET!"=="trainfp8"    goto :FP8TRAIN
if /i "!PRESET!"=="test"        goto :TEST
if /i "!PRESET!"=="train"       goto :TEST

call :SHOW_PRESETS
echo.
echo Unknown preset: %~1
echo Normalized as: !PRESET!
exit /b 1

:SHOW_PRESETS
echo.
echo Available presets:
echo   start.bat default       - FP8 KV cache (no plugin)
echo   start.bat f16           - F16 KV cache (no plugin)
echo   start.bat smart         - F16 + Smart-KV balanced
echo   start.bat f16smart      - F16 + Smart-KV ultra
echo.
echo   FP8 + Smart-KV variants:
echo   start.bat fp8smart      - FP8 + Smart-KV ultra ^(max quality^)
echo   start.bat fp8balanced   - FP8 + Smart-KV balanced
echo   start.bat fp8light      - same as fp8balanced
echo   start.bat fp8aggressive - FP8 + Smart-KV ^(more tier 4 evictions^)
echo   start.bat fp8max        - FP8 + Smart-KV ultra, 128k ctx
echo   start.bat fp8train      - FP8 + Smart-KV train mode
echo.
echo   start.bat test          - F16 + Smart-KV training/export mode
goto :eof

:DEFAULT
echo [Preset: FP8 KV cache]
set "PRESET_ARGS=--cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup --plugin-kv-cache off --plugin-attn off"
goto :LAUNCH

:F16
echo [Preset: F16 KV cache]
set "PRESET_ARGS=--plugin-kv-cache off --plugin-attn off"
goto :LAUNCH

:SMART
echo [Preset: Smart-KV balanced]
set "SMART_KV_FP8=0"
set "SMART_KV_PROFILE=balanced"
set "PRESET_ARGS=--plugin-kv-cache on --plugin-attn off"
goto :LAUNCH

:F16SMART
echo [Preset: F16 + Smart-KV ultra]
set "SMART_KV_FP8=0"
set "SMART_KV_PROFILE=ultra"
set "PRESET_ARGS=--plugin-kv-cache on --plugin-attn off"
goto :LAUNCH

:FP8SMART
echo [Preset: FP8 + Smart-KV ultra]
set "SMART_KV_FP8=1"
set "SMART_KV_PROFILE=ultra"
set "FP8_KV_ENABLE=1"
set "PRESET_ARGS=--cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup --plugin-kv-cache on --plugin-attn off"
goto :LAUNCH

:FP8BALANCED
echo [Preset: FP8 + Smart-KV balanced]
set "SMART_KV_FP8=1"
set "SMART_KV_PROFILE=balanced"
set "FP8_KV_ENABLE=1"
set "PRESET_ARGS=--cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup --plugin-kv-cache on --plugin-attn off"
goto :LAUNCH

:FP8MAX
echo [Preset: FP8 + Smart-KV ultra, 128k context]
set "COMMON=-ngl 99 -fa on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0 --mirostat 0 -c 131072 -b 256 -ub 128 --parallel 1 --host 0.0.0.0 --port 12345"
set "SMART_KV_FP8=1"
set "SMART_KV_PROFILE=ultra"
set "FP8_KV_ENABLE=1"
set "PRESET_ARGS=--cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup --plugin-kv-cache on --plugin-attn off"
goto :LAUNCH

:FP8AGGRESSIVE
echo [Preset: FP8 + Smart-KV performance (aggressive eviction)]
set "SMART_KV_FP8=1"
set "SMART_KV_PROFILE=performance"
set "FP8_KV_ENABLE=1"
set "PRESET_ARGS=--cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup --plugin-kv-cache on --plugin-attn off"
goto :LAUNCH

:FP8TRAIN
echo [Preset: FP8 + Smart-KV training mode]
set "SMART_KV_FP8=1"
set "SMART_KV_TRAIN_MODE=1"
set "SMART_KV_SNAPSHOT_EXPORTS=1"
set "SMART_KV_PROFILE=balanced"
set "FP8_KV_ENABLE=1"
set "PRESET_ARGS=--cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --no-warmup --plugin-kv-cache on --plugin-attn off"
goto :LAUNCH

:TEST
echo [Preset: F16 + Smart-KV training mode]
set "SMART_KV_FP8=0"
set "SMART_KV_TRAIN_MODE=1"
set "SMART_KV_SNAPSHOT_EXPORTS=1"
set "SMART_KV_PROFILE=balanced"
set "PRESET_ARGS=--plugin-kv-cache on --plugin-attn off"
goto :LAUNCH

:LAUNCH
echo.
echo Server: %SERVER%
echo Preset: !PRESET!
echo.
"%SERVER%" -m "!MODEL!" !PRESET_ARGS! !COMMON!
exit /b !ERRORLEVEL!
