//
// Created by windy on 2023/8/30.
//

#ifndef CUDALEARNING_LOCAL_UTILS_H
#define CUDALEARNING_LOCAL_UTILS_H

#include <cuda_runtime.h>

void printDeviceProp(const cudaDeviceProp &prop) {
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %zu.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %zu.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %zu.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %zu.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %zu.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

#endif //CUDALEARNING_LOCAL_UTILS_H
