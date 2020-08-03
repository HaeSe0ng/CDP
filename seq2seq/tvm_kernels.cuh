#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void batch_matmul_1_2048_1_512_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute);
__global__ void batch_matmul_1_2048_1_256_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute);
__global__ void batch_matmul_1_16384_1_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute);
__global__ void batch_matmul_1_256_1_512_kernel(void *__restrict__ A,
                                                void *__restrict__ B,
                                                void *__restrict__ compute);
__global__ void batch_matmul_64_2048_1_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute);
__global__ void batch_matmul_1_1_2048_512_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute);
__global__ void batch_matmul_1_1_16384_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute);
__global__ void batch_matmul_1_4_2048_512_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute);
__global__ void batch_matmul_1_4_16384_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute);
__global__ void batch_matmul_1_16_2048_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute);
__global__ void batch_matmul_1_16_16384_512_kernel(void *__restrict__ A,
                                                   void *__restrict__ B,
                                                   void *__restrict__ compute);
__global__ void batch_matmul_1_64_2048_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute);
__global__ void batch_matmul_1_64_16384_512_kernel(void *__restrict__ A,
                                                   void *__restrict__ B,
                                                   void *__restrict__ compute);

typedef void (*kern_func)(void *, void *, void *);

__host__ __device__ kern_func _matmul_kernel_launch_cfg(
    int bsz, int M, int N, int K, dim3 *gridDim, dim3 *blockDim);
struct matmul_kernel_launch_cfg
{
  kern_func func;
  dim3 gridDim;
  dim3 blockDim;
  __host__ __device__ matmul_kernel_launch_cfg(int bsz, int M, int N, int K)
  {
    func = _matmul_kernel_launch_cfg(bsz, M, N, K, &gridDim, &blockDim);
  }
};