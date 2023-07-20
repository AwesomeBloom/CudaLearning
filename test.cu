#include <stdio.h>

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("GPU Architecture: sm_%d%d\n", prop.major, prop.minor);

    return 0;
}
