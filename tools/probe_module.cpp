/// probe_module — try hipModuleLoad to get detailed error
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    int ndev;
    hipGetDeviceCount(&ndev);
    for (int d = 0; d < ndev; d++) {
        hipDeviceProp_t p;
        hipGetDeviceProperties(&p, d);
        printf("[%d] %s  arch=%s\n", d, p.name, p.gcnArchName);
        if (strstr(p.gcnArchName, "gfx12")) {
            hipSetDevice(d);
            printf("Set device %d\n", d);
            // Try loading a dummy code object from file
            // Check if the hip runtime reports capability
            int attr;
            hipError_t e = hipDeviceGetAttribute(&attr, hipDeviceAttributeComputeCapabilityMajor, d);
            printf("  hipDeviceGetAttribute: %s (major=%d)\n", hipGetErrorString(e), attr);
            e = hipDeviceGetAttribute(&attr, hipDeviceAttributeComputeCapabilityMinor, d);
            printf("  minor=%d\n", attr);
            // Check if kernels can be launched at all
            int can_access;
            e = hipDeviceGetAttribute(&can_access, hipDeviceAttributeCanLaunchKernel, d);
            printf("  canLaunchKernel: %s (val=%d)\n", hipGetErrorString(e), can_access);
            // Check cooperative launch support
            e = hipDeviceGetAttribute(&can_access, hipDeviceAttributeCooperativeLaunch, d);
            printf("  cooperativeLaunch: %s (val=%d)\n", hipGetErrorString(e), can_access);
            break;
        }
    }
    return 0;
}
