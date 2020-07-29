#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void
batch_matmul_1_2048_1_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                 void *__restrict__ compute);
__global__ void
batch_matmul_1_2048_1_256_kernel(void *__restrict__ A, void *__restrict__ B,
                                 void *__restrict__ compute);
__global__ void
batch_matmul_1_16384_1_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                  void *__restrict__ compute);
__global__ void
batch_matmul_1_256_1_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                void *__restrict__ compute);
__global__ void
batch_matmul_64_2048_1_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                  void *__restrict__ compute);
__global__ void
batch_matmul_1_1_2048_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                 void *__restrict__ compute);
__global__ void
batch_matmul_1_1_16384_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                  void *__restrict__ compute);
__global__ void
batch_matmul_1_4_2048_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                 void *__restrict__ compute);
__global__ void
batch_matmul_1_4_16384_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                  void *__restrict__ compute);
__global__ void
batch_matmul_1_64_2048_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                  void *__restrict__ compute);
__global__ void
batch_matmul_1_64_16384_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                   void *__restrict__ compute);
struct matmul_kernel_launch_cfg
{
  void (*func)(void *, void *, void *);
  dim3 gridDim;
  dim3 blockDim;
  __host__ __device__ matmul_kernel_launch_cfg(int bsz, int M, int N, int K)
  {
    if (bsz == 1 && M == 2048 && N == 1 && K == 512)
    {
      gridDim = dim3(1, 256, 1);
      blockDim = dim3(1, 2, 1);
      func = batch_matmul_1_2048_1_512_kernel;
    }
    else if (bsz == 1 && M == 16384 && N == 1 && K == 512)
    {
      gridDim = dim3(1, 256, 1);
      blockDim = dim3(1, 16, 1);
      func = batch_matmul_1_16384_1_512_kernel;
    }
    else if (bsz == 1 && M == 256 && N == 1 && K == 512)
    {
      gridDim = dim3(1, 4, 1);
      blockDim = dim3(1, 32, 1);
      func = batch_matmul_1_256_1_512_kernel;
    }
    else if (bsz == 64 && M == 2048 && N == 1 && K == 512)
    {
      gridDim = dim3(1, 256, 1);
      blockDim = dim3(1, 2, 1);
      func = batch_matmul_64_2048_1_512_kernel;
    }
    else if (bsz == 1 && M == 1 && N == 2048 && K == 512)
    {
      gridDim = dim3(64, 1, 1);
      blockDim = dim3(8, 1, 1);
      func = batch_matmul_1_1_2048_512_kernel;
    }
    else if (bsz == 1 && M == 1 && N == 16384 && K == 512)
    {
      gridDim = dim3(512, 1, 1);
      blockDim = dim3(32, 1, 1);
      func = batch_matmul_1_1_16384_512_kernel;
    }
    else if (bsz == 1 && M == 4 && N == 2048 && K == 512)
    {
      gridDim = dim3(256, 1, 1);
      blockDim = dim3(8, 4, 1);
      func = batch_matmul_1_4_2048_512_kernel;
    }
    else if (bsz == 1 && M == 4 && N == 16384 && K == 512)
    {
      gridDim = dim3(256, 1, 1);
      blockDim = dim3(32, 1, 1);
      func = batch_matmul_1_4_16384_512_kernel;
    }
    else if (bsz == 1 && M == 64 && N == 2048 && K == 512)
    {
      gridDim = dim3(64, 2, 1);
      blockDim = dim3(16, 2, 1);
      func = batch_matmul_1_64_2048_512_kernel;
    }
    else if (bsz == 1 && M == 64 && N == 16384 && K == 512)
    {
      gridDim = dim3(256, 1, 1);
      blockDim = dim3(32, 2, 1);
      func = batch_matmul_1_64_16384_512_kernel;
    }
    else
    {
      printf("batch_matmul: WRONG ARGS (bsz=%d, M=%d, N=%d, K=%d)", bsz, M, N, K);
    }
  }
};