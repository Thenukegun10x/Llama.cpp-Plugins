/// minimal_test — test basic HIP kernel launch on gfx1201
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(call) do { \
    hipError_t e = call; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__global__ void k_add(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    c[i] = a[i] + b[i];
}

int main() {
    int ndev;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    for (int d = 0; d < ndev; d++) {
        hipDeviceProp_t p;
        HIP_CHECK(hipGetDeviceProperties(&p, d));
        printf("[%d] %s  arch=%s\n", d, p.name, p.gcnArchName);
    }
    int target = -1;
    for (int d = 0; d < ndev; d++) {
        hipDeviceProp_t p;
        HIP_CHECK(hipGetDeviceProperties(&p, d));
        if (strstr(p.gcnArchName, "gfx12")) { target = d; break; }
    }
    if (target < 0) { printf("no gfx12 device\n"); return 1; }
    HIP_CHECK(hipSetDevice(target));
    printf("Using device %d\n", target);

    const int N = 1024;
    float *a, *b, *c;
    HIP_CHECK(hipMalloc(&a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&b, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&c, N * sizeof(float)));
    float ha[N], hb[N], hc[N];
    for (int i = 0; i < N; i++) { ha[i] = i; hb[i] = i * 2; }
    HIP_CHECK(hipMemcpy(a, ha, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b, hb, N * sizeof(float), hipMemcpyHostToDevice));
    k_add<<<N/256, 256>>>(a, b, c, N);
    hipError_t err = hipGetLastError();
    printf("kernel launch error: %s\n", hipGetErrorString(err));
    HIP_CHECK(hipMemcpy(hc, c, N * sizeof(float), hipMemcpyDeviceToHost));
    int ok = 1;
    for (int i = 0; i < N; i++) if (hc[i] != ha[i] + hb[i]) { ok = 0; break; }
    printf("test: %s\n", ok ? "PASS" : "FAIL");
    HIP_CHECK(hipFree(a)); HIP_CHECK(hipFree(b)); HIP_CHECK(hipFree(c));
    printf("HIP runtime version: %d\n", HIP_VERSION);
    printf("HIP version major: %d minor: %d\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR);
    return 0;
}
