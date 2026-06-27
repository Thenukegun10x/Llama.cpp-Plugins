/// probe device features
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

int main() {
    int ndev;
    hipGetDeviceCount(&ndev);
    for (int d = 0; d < ndev; d++) {
        hipDeviceProp_t p;
        hipGetDeviceProperties(&p, d);
        printf("[%d] %s  arch=%s\n", d, p.name, p.gcnArchName);
    }
    // Check the gfx12 device
    for (int d = 0; d < ndev; d++) {
        hipDeviceProp_t p;
        hipGetDeviceProperties(&p, d);
        if (strstr(p.gcnArchName, "gfx12")) {
            printf("\nDevice %d features:\n", d);
            printf("  arch.hasGlobalInt32Atomics:  %d\n", p.arch.hasGlobalInt32Atomics);
            printf("  arch.hasSharedInt32Atomics:  %d\n", p.arch.hasSharedInt32Atomics);
            printf("  arch.hasDoubles:             %d\n", p.arch.hasDoubles);
            printf("  arch.hasWarpVote:            %d\n", p.arch.hasWarpVote);
            printf("  arch.hasWarpBallot:          %d\n", p.arch.hasWarpBallot);
            printf("  arch.hasWarpShuffle:         %d\n", p.arch.hasWarpShuffle);
            printf("  arch.hasFunnelShift:         %d\n", p.arch.hasFunnelShift);
            printf("  arch.hasThreadFenceSystem:   %d\n", p.arch.hasThreadFenceSystem);
            printf("  arch.hasSyncThreadsExt:      %d\n", p.arch.hasSyncThreadsExt);
            printf("  arch.hasSurfaceFuncs:        %d\n", p.arch.hasSurfaceFuncs);
            printf("  arch.has3dGrid:              %d\n", p.arch.has3dGrid);
            printf("  arch.hasDynamicParallelism:  %d\n", p.arch.hasDynamicParallelism);
            printf("  warpSize:                    %d\n", p.warpSize);
            printf("  major:                       %d\n", p.major);
            printf("  minor:                       %d\n", p.minor);
            printf("  integrated:                  %d\n", p.integrated);
            printf("  cooperativeLaunch:           %d\n", p.cooperativeLaunch);
            printf("  isLargeBar:                  %d\n", p.isLargeBar);
            break;
        }
    }
    return 0;
}
