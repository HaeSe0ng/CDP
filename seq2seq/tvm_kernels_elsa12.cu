#include "tvm_kernels.cuh"

#include <stdio.h>

#include <cstdlib>

#include "seq2seq.h"
#include "util.h"
// batch_matmul_bsz_M_N_K

__host__ __device__ kern_func _matmul_kernel_launch_cfg(int M, int N, int K,
                                                        dim3 *gridDim,
                                                        dim3 *blockDim) {
  kern_func func;
  if (M == 1 && N == 2048 && K == 512) {
    *gridDim = dim3(2048, 1, 1);
    *blockDim = dim3(32, 1, 1);
    func = matmul_1_2048_512_kernel;
  } else if (M == 1 && N == 16384 && K == 512) {
    *gridDim = dim3(16384, 1, 1);
    *blockDim = dim3(64, 1, 1);
    func = matmul_1_16384_512_kernel;
  } else if (M == 4 && N == 2048 && K == 512) {
    *gridDim = dim3(2048, 4, 1);
    *blockDim = dim3(32, 1, 1);
    func = matmul_4_2048_512_kernel;
  } else if (M == 4 && N == 16384 && K == 512) {
    *gridDim = dim3(16384, 4, 1);
    *blockDim = dim3(64, 1, 1);
    func = matmul_4_16384_512_kernel;
  } else if (M == 16 && N == 2048 && K == 512) {
    *gridDim = dim3(2048, 16, 1);
    *blockDim = dim3(32, 1, 1);
    func = matmul_16_2048_512_kernel;
  } else if (M == 16 && N == 16384 && K == 512) {
    *gridDim = dim3(16384, 16, 1);
    *blockDim = dim3(128, 1, 1);
    func = matmul_16_16384_512_kernel;
  } else if (M == 64 && N == 2048 && K == 512) {
    *gridDim = dim3(2048, 64, 1);
    *blockDim = dim3(32, 1, 1);
    func = matmul_64_2048_512_kernel;
  } else if (M == 64 && N == 16384 && K == 512) {
    *gridDim = dim3(16384, 64, 1);
    *blockDim = dim3(64, 1, 1);
    func = matmul_64_16384_512_kernel;
  } else {
    printf("batch_matmul: WRONG ARGS (M=%d, N=%d, K=%d)", M, N, K);
  }
  return func;
}

__global__ void matmul_1_2048_512_kernel(void *__restrict__ A,
                                         void *__restrict__ B,
                                         void *__restrict__ T_dense) {
  float T_dense_rf[1];
  float red_buf0[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    T_dense_rf[(0)] =
        (T_dense_rf[(0)] +
         (((float *)A)[(((k_outer * 32) + ((int)threadIdx.x)))] *
          ((float *)B)[((((((int)blockIdx.x) * 512) + (k_outer * 32)) +
                         ((int)threadIdx.x)))]));
  }
  unsigned int mask[1];
  float t0[1];
  red_buf0[(0)] = T_dense_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    ((float *)T_dense)[(((int)blockIdx.x))] = red_buf0[(0)];
  }
}

__global__ void matmul_1_16384_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ T_dense) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    T_dense_rf[(0)] =
        (T_dense_rf[(0)] +
         (((float *)A)[(((k_outer * 64) + ((int)threadIdx.x)))] *
          ((float *)B)[((((((int)blockIdx.x) * 512) + (k_outer * 64)) +
                         ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float *)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    ((float *)T_dense)[(((int)blockIdx.x))] = ((volatile float *)red_buf0)[(0)];
  }
}

__global__ void matmul_4_2048_512_kernel(void *__restrict__ A,
                                         void *__restrict__ B,
                                         void *__restrict__ T_dense) {
  float T_dense_rf[1];
  float red_buf0[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    T_dense_rf[(0)] =
        (T_dense_rf[(0)] +
         (((float *)A)[((((((int)blockIdx.y) * 512) + (k_outer * 32)) +
                         ((int)threadIdx.x)))] *
          ((float *)B)[((((((int)blockIdx.x) * 512) + (k_outer * 32)) +
                         ((int)threadIdx.x)))]));
  }
  unsigned int mask[1];
  float t0[1];
  red_buf0[(0)] = T_dense_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    ((float *)T_dense)[(((((int)blockIdx.y) * 2048) + ((int)blockIdx.x)))] =
        red_buf0[(0)];
  }
}

__global__ void matmul_4_16384_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ T_dense) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    T_dense_rf[(0)] =
        (T_dense_rf[(0)] +
         (((float *)A)[((((((int)blockIdx.y) * 512) + (k_outer * 64)) +
                         ((int)threadIdx.x)))] *
          ((float *)B)[((((((int)blockIdx.x) * 512) + (k_outer * 64)) +
                         ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float *)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    ((float *)T_dense)[(((((int)blockIdx.y) * 16384) + ((int)blockIdx.x)))] =
        ((volatile float *)red_buf0)[(0)];
  }
}

__global__ void matmul_16_2048_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ T_dense) {
  float T_dense_rf[1];
  float red_buf0[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    T_dense_rf[(0)] =
        (T_dense_rf[(0)] +
         (((float *)A)[((((((int)blockIdx.y) * 512) + (k_outer * 32)) +
                         ((int)threadIdx.x)))] *
          ((float *)B)[((((((int)blockIdx.x) * 512) + (k_outer * 32)) +
                         ((int)threadIdx.x)))]));
  }
  unsigned int mask[1];
  float t0[1];
  red_buf0[(0)] = T_dense_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    ((float *)T_dense)[(((((int)blockIdx.y) * 2048) + ((int)blockIdx.x)))] =
        red_buf0[(0)];
  }
}

__global__ void matmul_16_16384_512_kernel(void *__restrict__ A,
                                           void *__restrict__ B,
                                           void *__restrict__ T_dense) {
  float T_dense_rf[1];
  __shared__ float red_buf0[128];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 4; ++k_outer) {
    T_dense_rf[(0)] =
        (T_dense_rf[(0)] +
         (((float *)A)[((((((int)blockIdx.y) * 512) + (k_outer * 128)) +
                         ((int)threadIdx.x)))] *
          ((float *)B)[((((((int)blockIdx.x) * 512) + (k_outer * 128)) +
                         ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float *)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 64))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    ((float *)T_dense)[(((((int)blockIdx.y) * 16384) + ((int)blockIdx.x)))] =
        ((volatile float *)red_buf0)[(0)];
  }
}

__global__ void matmul_64_2048_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ T_dense) {
  float T_dense_rf[1];
  float red_buf0[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    T_dense_rf[(0)] =
        (T_dense_rf[(0)] +
         (((float *)A)[((((((int)blockIdx.y) * 512) + (k_outer * 32)) +
                         ((int)threadIdx.x)))] *
          ((float *)B)[((((((int)blockIdx.x) * 512) + (k_outer * 32)) +
                         ((int)threadIdx.x)))]));
  }
  unsigned int mask[1];
  float t0[1];
  red_buf0[(0)] = T_dense_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    ((float *)T_dense)[(((((int)blockIdx.y) * 2048) + ((int)blockIdx.x)))] =
        red_buf0[(0)];
  }
}

__global__ void matmul_64_16384_512_kernel(void *__restrict__ A,
                                           void *__restrict__ B,
                                           void *__restrict__ T_dense) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    T_dense_rf[(0)] =
        (T_dense_rf[(0)] +
         (((float *)A)[((((((int)blockIdx.y) * 512) + (k_outer * 64)) +
                         ((int)threadIdx.x)))] *
          ((float *)B)[((((((int)blockIdx.x) * 512) + (k_outer * 64)) +
                         ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float *)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float *)red_buf0)[(((int)threadIdx.x))] =
        (((volatile float *)red_buf0)[(((int)threadIdx.x))] +
         ((volatile float *)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    ((float *)T_dense)[(((((int)blockIdx.y) * 16384) + ((int)blockIdx.x)))] =
        ((volatile float *)red_buf0)[(0)];
  }
}
