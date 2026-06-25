@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

rem Use short (8.3) paths to avoid space-in-path issues
for %%A in ("%~dp0..\..\llama.cpp") do set LLAMA_SHORT=%%~sA
for %%A in ("%~dp0..\turbo-quant-plugin") do set TURBO_SHORT=%%~sA
for %%A in ("%~dp0..\..\SageAttention-rocm\plugin-system") do set SYS_SHORT=%%~sA
for %%A in ("%~dp0.") do set PLUGIN_SHORT=%%~sA
for %%A in ("%~dp0build") do set BUILD_SHORT=%%~sA

if not exist "%BUILD_SHORT%" mkdir "%BUILD_SHORT%"

set CC="C:\Program Files\AMD\ROCm\7.1\bin\clang++.exe"

echo Building smart-tq-plugin...

%CC% -shared -O2 -DNDEBUG ^
    "%PLUGIN_SHORT%\smart-tq-plugin.cpp" ^
    "%PLUGIN_SHORT%\smart-kv-cache.c" ^
    "%TURBO_SHORT%\turboquant.cpp" ^
    -I"%PLUGIN_SHORT%" ^
    -I"%TURBO_SHORT%" ^
    -I"%LLAMA_SHORT%\ggml\include" ^
    -I"%SYS_SHORT%" ^
    -o "%BUILD_SHORT%\smart-tq-plugin.dll"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS
    dir /b "%BUILD_SHORT%\smart-tq-plugin.dll"
    copy /Y "%BUILD_SHORT%\smart-tq-plugin.dll" "%PLUGIN_SHORT%\smart-tq-plugin.dll" >nul
    if !ERRORLEVEL! EQU 0 (
        echo Updated %PLUGIN_SHORT%\smart-tq-plugin.dll
    ) else (
        echo Failed to update %PLUGIN_SHORT%\smart-tq-plugin.dll
        exit /b !ERRORLEVEL!
    )
) else (
    echo.
    echo BUILD FAILED with code %ERRORLEVEL%
)

exit /b %ERRORLEVEL%
