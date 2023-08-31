#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_kernel(int a, int b, int *c) {
    *c = a + b;
}

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

__host__ void perform_add() {
    int res = 0;
    int *res_device;

    cudaMalloc( (void**) &res_device, sizeof (int));

    add_kernel<<<1, 1>>>(1, 4, res_device);

    cudaMemcpy(&res, res_device, sizeof (int), cudaMemcpyDeviceToHost);

    cudaFree(res_device);

    printf("1 + 4 = %d\n", res);
}

__host__ void perform_device_query() {
    int count = 0;

    cudaGetDeviceCount(&count);

    printf("device count: %d\n", count);

    // get the last device
    if (count == 0) {
        printf("There's no cuda device!\n");
    } else {
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp curProp{};
            if (cudaGetDeviceProperties(&curProp, i) == cudaSuccess) {
                printDeviceProp(curProp);
            }
        }
    }
}

int main() {

    perform_add();

    perform_device_query();

    return 0;
}