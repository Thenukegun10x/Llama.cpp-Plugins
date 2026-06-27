@echo off
REM Build the WMMA FP8 verification script for gfx1201 (RDNA4)
REM Adjust ROCM_PATH if your ROCm is installed elsewhere

set ROCM_PATH=C:\Program Files\AMD\ROCm
if not exist "%ROCM_PATH%" set ROCM_PATH=C:\HIP
if not exist "%ROCM_PATH%" set ROCM_PATH=%HIP_PATH%

set HIPCC=%ROCM_PATH%\bin\hipcc
if not exist "%HIPCC%" (
    echo hipcc not found at %HIPCC%
    echo Set ROCM_PATH or HIP_PATH to your ROCm installation
    exit /b 1
)

echo Using HIPCC: %HIPCC%
echo Building wmma_fp8_verify for gfx1201...

"%HIPCC%" ^
    -O3 ^
    --offload-arch=gfx1201 ^
    -o wmma_fp8_verify.exe ^
    wmma_fp8_verify.cpp ^
    -D__HIP_PLATFORM_AMD__

if %ERRORLEVEL% equ 0 (
    echo Build successful: wmma_fp8_verify.exe
) else (
    echo Build FAILED
    exit /b %ERRORLEVEL%
)
