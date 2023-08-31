cmake_minimum_required(VERSION 3.25)
project(CudaLearning CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2")
add_executable(CudaLearning
        main.cu
        local_utils.h)

set_target_properties(CudaLearning PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
