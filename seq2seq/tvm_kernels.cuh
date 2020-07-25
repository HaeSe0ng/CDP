#pragma once
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