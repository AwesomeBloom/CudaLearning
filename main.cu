#include <cstdio>
#include <cuda_runtime.h>
#include "control_config.h"
#include "utils.h"

namespace basic {

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
}

namespace serial {
    [[maybe_unused]] __global__ void serial_vector_sum(const int *a, const int *b, int *result, size_t len) {
        for (int i = 0; i < len; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}

namespace parallel {
    const int parallelism = 1024;
    __global__ void vector_sum(const int *a, const int *b, int *result, size_t len) {
        unsigned int tid = blockIdx.x;

        while(tid < len) {
            if (result && a && b) {
                result[tid] = a[tid] + b[tid];
                tid += parallelism;
            }
        }
    }

    __host__ void perform_vector_sum() {
        const size_t vector_size = 800000000ULL;

        int *a = new int[vector_size], *b = new int[vector_size], *result = new int[vector_size];

        // initialize a & b
        for (int i = 0; i < vector_size; ++i) {
            a[i] = -i;
            b[i] = i * i;
        }

        int *a_device = nullptr, *b_device = nullptr, *result_device = nullptr;

        size_t whole_size = vector_size * sizeof (int);
        cudaMalloc( (void**)&a_device, whole_size);
        cudaMalloc( (void**)&b_device, whole_size);
        cudaMalloc( (void**)&result_device, whole_size);

        cudaMemcpy( a_device, a, whole_size, cudaMemcpyHostToDevice);
        cudaMemcpy( b_device, b, whole_size, cudaMemcpyHostToDevice);

        vector_sum<<<parallelism, 1>>>(a_device, b_device, result_device, vector_size);

        cudaMemcpy(result, result_device, whole_size, cudaMemcpyDeviceToHost);

        cudaFree(a_device);
        cudaFree(b_device);
        cudaFree(result_device);

        free(a);
        free(b);

        for (int i = 0; i < 10; ++i) {
            printf("result[%d] = %d\n", i, result[i]);
        }
        free(result);

    }

}

int main() {

    if(config::using_basic) {
        basic::perform_add();
        basic::perform_device_query();
    }

    if (config::using_parallel) {
        auto exec_time = get_executing_time<double>(parallel::perform_vector_sum);
        printf("parallel time: %.2lfms. ", exec_time);
    }

    return 0;
}