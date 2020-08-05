#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_1_2048_512_kernel(void *__restrict__ A,
                                         void *__restrict__ B,
                                         void *__restrict__ T_dense);
__global__ void matmul_1_16384_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ T_dense);
__global__ void matmul_4_2048_512_kernel(void *__restrict__ A,
                                         void *__restrict__ B,
                                         void *__restrict__ T_dense);
__global__ void matmul_4_16384_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ T_dense);
__global__ void matmul_16_2048_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ T_dense);
__global__ void matmul_16_16384_512_kernel(void *__restrict__ A,
                                           void *__restrict__ B,
                                           void *__restrict__ T_dense);
__global__ void matmul_64_2048_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ T_dense);
__global__ void matmul_64_16384_512_kernel(void *__restrict__ A,
                                           void *__restrict__ B,
                                           void *__restrict__ T_dense);

typedef void (*kern_func)(void *, void *, void *);

__host__ __device__ kern_func _matmul_kernel_launch_cfg(
    int M, int N, int K, dim3 *gridDim, dim3 *blockDim);
struct matmul_kernel_launch_cfg
{
  kern_func func;
  dim3 gridDim;
  dim3 blockDim;
  __host__ __device__ matmul_kernel_launch_cfg(int M, int N, int K)
  {
    func = _matmul_kernel_launch_cfg(M, N, K, &gridDim, &blockDim);
  }
};