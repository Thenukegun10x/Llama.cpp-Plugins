#!/bin/bash
# Build wmma_fp8_verify for gfx1201 (RDNA4)
set -e

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/bin/hipcc"
ROCWMMA="${ROCM_PATH}/rocwmma"

if [ ! -f "${HIPCC}" ]; then
    echo "hipcc not found at ${HIPCC}"
    echo "Set ROCM_PATH to your ROCm installation"
    exit 1
fi

echo "Using: ${HIPCC}"
echo "Building wmma_fp8_verify for gfx1201..."

"${HIPCC}" \
    -O3 \
    --offload-arch=gfx1201 \
    -I"${ROCM_PATH}/include" \
    -L"${ROCM_PATH}/lib" \
    -lrocwmma \
    -o wmma_fp8_verify \
    wmma_fp8_verify.cpp

echo "OK: wmma_fp8_verify"
