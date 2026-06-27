@echo off
setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set FP8_PLUGIN_DIR=%SCRIPT_DIR%
set GGML_INCLUDE=%SCRIPT_DIR%..\..\llama.cpp\ggml\include
set BUILD_DIR=%FP8_PLUGIN_DIR%build

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo [fp8-kv-plugin] Building release DLL...
cl /nologo /O2 /LD /I"%GGML_INCLUDE%" "%FP8_PLUGIN_DIR%fp8-kv-plugin.cpp" /Fe:"%BUILD_DIR%fp8-kv-plugin.dll"

if %ERRORLEVEL% equ 0 (
    echo [fp8-kv-plugin] BUILD SUCCESS: %BUILD_DIR%fp8-kv-plugin.dll
) else (
    echo [fp8-kv-plugin] BUILD FAILED
    exit /b 1
)
