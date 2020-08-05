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
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(8, 4, 1);
    func = matmul_4_2048_512_kernel;
  } else if (M == 4 && N == 16384 && K == 512) {
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(32, 1, 1);
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
    *gridDim = dim3(1, 16, 1);
    *blockDim = dim3(16, 16, 1);
    func = matmul_64_2048_512_kernel;
  } else if (M == 64 && N == 16384 && K == 512) {
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(32, 2, 1);
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
                                         void *__restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[32];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    A_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] = ((float *)A)[(
        (((((int)threadIdx.y) * 512) + (k_outer * 8)) + ((int)threadIdx.x)))];
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
      B_shared[((((((int)threadIdx.y) * 16) + (ax1_inner * 8)) +
                 ((int)threadIdx.x)))] =
          ((float *)B)[(
              (((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 1024)) +
                 (ax1_inner * 512)) +
                (k_outer * 8)) +
               ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 8) + k_inner))];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 8) + k_inner))];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[((((((int)threadIdx.y) * 2048) + (((int)blockIdx.x) * 8)) +
                       ((int)threadIdx.x)))] = compute_local[(0)];
}

__global__ void matmul_4_16384_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ compute) {
  float compute_local[8];
  __shared__ float A_shared[256];
  __shared__ float B_shared[4096];
  float A_shared_local[4];
  float B_shared_local[2];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      compute_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
#pragma unroll
      for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
        A_shared[(
            (((ax1_inner * 64) + (((int)threadIdx.x) * 2)) + ax2_inner))] =
            ((float *)A)[(((((ax1_inner * 512) + (k_outer * 64)) +
                            (((int)threadIdx.x) * 2)) +
                           ax2_inner))];
      }
    }
    for (int ax1_inner1 = 0; ax1_inner1 < 64; ++ax1_inner1) {
#pragma unroll
      for (int ax2_inner1 = 0; ax2_inner1 < 2; ++ax2_inner1) {
        B_shared[(
            (((ax1_inner1 * 64) + (((int)threadIdx.x) * 2)) + ax2_inner1))] =
            ((float *)
                 B)[((((((((int)blockIdx.x) * 32768) + (ax1_inner1 * 512)) +
                        (k_outer * 64)) +
                       (((int)threadIdx.x) * 2)) +
                      ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 64; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        A_shared_local[(ax1)] = A_shared[(((ax1 * 64) + k_inner))];
      }
#pragma unroll
      for (int ax11 = 0; ax11 < 2; ++ax11) {
        B_shared_local[(ax11)] =
            B_shared[((((((int)threadIdx.x) * 128) + (ax11 * 64)) + k_inner))];
      }
#pragma unroll
      for (int i_c = 0; i_c < 4; ++i_c) {
#pragma unroll
        for (int j_c = 0; j_c < 2; ++j_c) {
          compute_local[(((i_c * 2) + j_c))] =
              (compute_local[(((i_c * 2) + j_c))] +
               (A_shared_local[(i_c)] * B_shared_local[(j_c)]));
        }
      }
    }
  }
#pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 4; ++i_inner_inner) {
#pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
      ((float *)
           compute)[(((((i_inner_inner * 16384) + (((int)blockIdx.x) * 64)) +
                       (((int)threadIdx.x) * 2)) +
                      j_inner_inner))] =
          compute_local[(((i_inner_inner * 2) + j_inner_inner))];
    }
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
  float T_dense_local[32];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[2048];
  float A_shared_local[4];
  float B_shared_local[8];
  float A_shared_local1[4];
  float B_shared_local1[8];
  T_dense_local[(0)] = 0.000000e+00f;
  T_dense_local[(4)] = 0.000000e+00f;
  T_dense_local[(8)] = 0.000000e+00f;
  T_dense_local[(12)] = 0.000000e+00f;
  T_dense_local[(16)] = 0.000000e+00f;
  T_dense_local[(20)] = 0.000000e+00f;
  T_dense_local[(24)] = 0.000000e+00f;
  T_dense_local[(28)] = 0.000000e+00f;
  T_dense_local[(1)] = 0.000000e+00f;
  T_dense_local[(5)] = 0.000000e+00f;
  T_dense_local[(9)] = 0.000000e+00f;
  T_dense_local[(13)] = 0.000000e+00f;
  T_dense_local[(17)] = 0.000000e+00f;
  T_dense_local[(21)] = 0.000000e+00f;
  T_dense_local[(25)] = 0.000000e+00f;
  T_dense_local[(29)] = 0.000000e+00f;
  T_dense_local[(2)] = 0.000000e+00f;
  T_dense_local[(6)] = 0.000000e+00f;
  T_dense_local[(10)] = 0.000000e+00f;
  T_dense_local[(14)] = 0.000000e+00f;
  T_dense_local[(18)] = 0.000000e+00f;
  T_dense_local[(22)] = 0.000000e+00f;
  T_dense_local[(26)] = 0.000000e+00f;
  T_dense_local[(30)] = 0.000000e+00f;
  T_dense_local[(3)] = 0.000000e+00f;
  T_dense_local[(7)] = 0.000000e+00f;
  T_dense_local[(11)] = 0.000000e+00f;
  T_dense_local[(15)] = 0.000000e+00f;
  T_dense_local[(19)] = 0.000000e+00f;
  T_dense_local[(23)] = 0.000000e+00f;
  T_dense_local[(27)] = 0.000000e+00f;
  T_dense_local[(31)] = 0.000000e+00f;
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 8) {
        A_shared[(((((((int)threadIdx.y) * 32) + (ax0_inner * 8)) +
                    (((int)threadIdx.x) * 4)) +
                   ax1_inner_inner))] =
            ((float *)A)[(((((((int)threadIdx.y) * 2048) + (ax0_inner * 512)) +
                            (((int)threadIdx.x) * 4)) +
                           ax1_inner_inner))];
      }
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 8; ++ax0_inner1) {
    for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner1) < 8) {
        B_shared[(((((((int)threadIdx.y) * 64) + (ax0_inner1 * 8)) +
                    (((int)threadIdx.x) * 4)) +
                   ax1_inner_inner1))] =
            ((float *)B)[(
                (((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 4096)) +
                   (ax0_inner1 * 512)) +
                  (((int)threadIdx.x) * 4)) +
                 ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 63; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 4; ++ax0_inner2) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 8) {
          if ((((k_outer_outer * 8) + (((int)threadIdx.x) * 4)) +
               ax1_inner_inner2) < 504) {
            A_shared[((((((((k_outer_outer + 1) & 1) * 512) +
                          (((int)threadIdx.y) * 32)) +
                         (ax0_inner2 * 8)) +
                        (((int)threadIdx.x) * 4)) +
                       ax1_inner_inner2))] =
                ((float *)A)[(
                    ((((((((int)threadIdx.y) * 2048) + (ax0_inner2 * 512)) +
                        (k_outer_outer * 8)) +
                       (((int)threadIdx.x) * 4)) +
                      ax1_inner_inner2) +
                     8))];
          }
        }
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 8; ++ax0_inner3) {
      for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner3) < 8) {
          if ((((k_outer_outer * 8) + (((int)threadIdx.x) * 4)) +
               ax1_inner_inner3) < 504) {
            B_shared[((((((((k_outer_outer + 1) & 1) * 1024) +
                          (((int)threadIdx.y) * 64)) +
                         (ax0_inner3 * 8)) +
                        (((int)threadIdx.x) * 4)) +
                       ax1_inner_inner3))] =
                ((float *)B)[((((((((((int)blockIdx.y) * 65536) +
                                    (((int)threadIdx.y) * 4096)) +
                                   (ax0_inner3 * 512)) +
                                  (k_outer_outer * 8)) +
                                 (((int)threadIdx.x) * 4)) +
                                ax1_inner_inner3) +
                               8))];
          }
        }
      }
    }
    A_shared_local[(0)] =
        A_shared[((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)))];
    A_shared_local[(1)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 128))];
    A_shared_local[(2)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 256))];
    A_shared_local[(3)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 384))];
    B_shared_local[(0)] =
        B_shared[((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)))];
    B_shared_local[(1)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 128))];
    B_shared_local[(2)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 256))];
    B_shared_local[(3)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 384))];
    B_shared_local[(4)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 512))];
    B_shared_local[(5)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 640))];
    B_shared_local[(6)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 768))];
    B_shared_local[(7)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 896))];
    T_dense_local[(0)] =
        (T_dense_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    T_dense_local[(4)] =
        (T_dense_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    T_dense_local[(8)] =
        (T_dense_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
    T_dense_local[(12)] =
        (T_dense_local[(12)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    T_dense_local[(16)] =
        (T_dense_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
    T_dense_local[(20)] =
        (T_dense_local[(20)] + (A_shared_local[(0)] * B_shared_local[(5)]));
    T_dense_local[(24)] =
        (T_dense_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
    T_dense_local[(28)] =
        (T_dense_local[(28)] + (A_shared_local[(0)] * B_shared_local[(7)]));
    T_dense_local[(1)] =
        (T_dense_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
    T_dense_local[(5)] =
        (T_dense_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
    T_dense_local[(9)] =
        (T_dense_local[(9)] + (A_shared_local[(1)] * B_shared_local[(2)]));
    T_dense_local[(13)] =
        (T_dense_local[(13)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    T_dense_local[(17)] =
        (T_dense_local[(17)] + (A_shared_local[(1)] * B_shared_local[(4)]));
    T_dense_local[(21)] =
        (T_dense_local[(21)] + (A_shared_local[(1)] * B_shared_local[(5)]));
    T_dense_local[(25)] =
        (T_dense_local[(25)] + (A_shared_local[(1)] * B_shared_local[(6)]));
    T_dense_local[(29)] =
        (T_dense_local[(29)] + (A_shared_local[(1)] * B_shared_local[(7)]));
    T_dense_local[(2)] =
        (T_dense_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
    T_dense_local[(6)] =
        (T_dense_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
    T_dense_local[(10)] =
        (T_dense_local[(10)] + (A_shared_local[(2)] * B_shared_local[(2)]));
    T_dense_local[(14)] =
        (T_dense_local[(14)] + (A_shared_local[(2)] * B_shared_local[(3)]));
    T_dense_local[(18)] =
        (T_dense_local[(18)] + (A_shared_local[(2)] * B_shared_local[(4)]));
    T_dense_local[(22)] =
        (T_dense_local[(22)] + (A_shared_local[(2)] * B_shared_local[(5)]));
    T_dense_local[(26)] =
        (T_dense_local[(26)] + (A_shared_local[(2)] * B_shared_local[(6)]));
    T_dense_local[(30)] =
        (T_dense_local[(30)] + (A_shared_local[(2)] * B_shared_local[(7)]));
    T_dense_local[(3)] =
        (T_dense_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    T_dense_local[(7)] =
        (T_dense_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    T_dense_local[(11)] =
        (T_dense_local[(11)] + (A_shared_local[(3)] * B_shared_local[(2)]));
    T_dense_local[(15)] =
        (T_dense_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
    T_dense_local[(19)] =
        (T_dense_local[(19)] + (A_shared_local[(3)] * B_shared_local[(4)]));
    T_dense_local[(23)] =
        (T_dense_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
    T_dense_local[(27)] =
        (T_dense_local[(27)] + (A_shared_local[(3)] * B_shared_local[(6)]));
    T_dense_local[(31)] =
        (T_dense_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
    A_shared_local[(0)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 1))];
    A_shared_local[(1)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 129))];
    A_shared_local[(2)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 257))];
    A_shared_local[(3)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 385))];
    B_shared_local[(0)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 1))];
    B_shared_local[(1)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 129))];
    B_shared_local[(2)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 257))];
    B_shared_local[(3)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 385))];
    B_shared_local[(4)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 513))];
    B_shared_local[(5)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 641))];
    B_shared_local[(6)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 769))];
    B_shared_local[(7)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 897))];
    T_dense_local[(0)] =
        (T_dense_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    T_dense_local[(4)] =
        (T_dense_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    T_dense_local[(8)] =
        (T_dense_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
    T_dense_local[(12)] =
        (T_dense_local[(12)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    T_dense_local[(16)] =
        (T_dense_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
    T_dense_local[(20)] =
        (T_dense_local[(20)] + (A_shared_local[(0)] * B_shared_local[(5)]));
    T_dense_local[(24)] =
        (T_dense_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
    T_dense_local[(28)] =
        (T_dense_local[(28)] + (A_shared_local[(0)] * B_shared_local[(7)]));
    T_dense_local[(1)] =
        (T_dense_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
    T_dense_local[(5)] =
        (T_dense_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
    T_dense_local[(9)] =
        (T_dense_local[(9)] + (A_shared_local[(1)] * B_shared_local[(2)]));
    T_dense_local[(13)] =
        (T_dense_local[(13)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    T_dense_local[(17)] =
        (T_dense_local[(17)] + (A_shared_local[(1)] * B_shared_local[(4)]));
    T_dense_local[(21)] =
        (T_dense_local[(21)] + (A_shared_local[(1)] * B_shared_local[(5)]));
    T_dense_local[(25)] =
        (T_dense_local[(25)] + (A_shared_local[(1)] * B_shared_local[(6)]));
    T_dense_local[(29)] =
        (T_dense_local[(29)] + (A_shared_local[(1)] * B_shared_local[(7)]));
    T_dense_local[(2)] =
        (T_dense_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
    T_dense_local[(6)] =
        (T_dense_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
    T_dense_local[(10)] =
        (T_dense_local[(10)] + (A_shared_local[(2)] * B_shared_local[(2)]));
    T_dense_local[(14)] =
        (T_dense_local[(14)] + (A_shared_local[(2)] * B_shared_local[(3)]));
    T_dense_local[(18)] =
        (T_dense_local[(18)] + (A_shared_local[(2)] * B_shared_local[(4)]));
    T_dense_local[(22)] =
        (T_dense_local[(22)] + (A_shared_local[(2)] * B_shared_local[(5)]));
    T_dense_local[(26)] =
        (T_dense_local[(26)] + (A_shared_local[(2)] * B_shared_local[(6)]));
    T_dense_local[(30)] =
        (T_dense_local[(30)] + (A_shared_local[(2)] * B_shared_local[(7)]));
    T_dense_local[(3)] =
        (T_dense_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    T_dense_local[(7)] =
        (T_dense_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    T_dense_local[(11)] =
        (T_dense_local[(11)] + (A_shared_local[(3)] * B_shared_local[(2)]));
    T_dense_local[(15)] =
        (T_dense_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
    T_dense_local[(19)] =
        (T_dense_local[(19)] + (A_shared_local[(3)] * B_shared_local[(4)]));
    T_dense_local[(23)] =
        (T_dense_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
    T_dense_local[(27)] =
        (T_dense_local[(27)] + (A_shared_local[(3)] * B_shared_local[(6)]));
    T_dense_local[(31)] =
        (T_dense_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
    A_shared_local[(0)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 2))];
    A_shared_local[(1)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 130))];
    A_shared_local[(2)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 258))];
    A_shared_local[(3)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 386))];
    B_shared_local[(0)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 2))];
    B_shared_local[(1)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 130))];
    B_shared_local[(2)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 258))];
    B_shared_local[(3)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 386))];
    B_shared_local[(4)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 514))];
    B_shared_local[(5)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 642))];
    B_shared_local[(6)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 770))];
    B_shared_local[(7)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 898))];
    T_dense_local[(0)] =
        (T_dense_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    T_dense_local[(4)] =
        (T_dense_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    T_dense_local[(8)] =
        (T_dense_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
    T_dense_local[(12)] =
        (T_dense_local[(12)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    T_dense_local[(16)] =
        (T_dense_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
    T_dense_local[(20)] =
        (T_dense_local[(20)] + (A_shared_local[(0)] * B_shared_local[(5)]));
    T_dense_local[(24)] =
        (T_dense_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
    T_dense_local[(28)] =
        (T_dense_local[(28)] + (A_shared_local[(0)] * B_shared_local[(7)]));
    T_dense_local[(1)] =
        (T_dense_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
    T_dense_local[(5)] =
        (T_dense_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
    T_dense_local[(9)] =
        (T_dense_local[(9)] + (A_shared_local[(1)] * B_shared_local[(2)]));
    T_dense_local[(13)] =
        (T_dense_local[(13)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    T_dense_local[(17)] =
        (T_dense_local[(17)] + (A_shared_local[(1)] * B_shared_local[(4)]));
    T_dense_local[(21)] =
        (T_dense_local[(21)] + (A_shared_local[(1)] * B_shared_local[(5)]));
    T_dense_local[(25)] =
        (T_dense_local[(25)] + (A_shared_local[(1)] * B_shared_local[(6)]));
    T_dense_local[(29)] =
        (T_dense_local[(29)] + (A_shared_local[(1)] * B_shared_local[(7)]));
    T_dense_local[(2)] =
        (T_dense_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
    T_dense_local[(6)] =
        (T_dense_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
    T_dense_local[(10)] =
        (T_dense_local[(10)] + (A_shared_local[(2)] * B_shared_local[(2)]));
    T_dense_local[(14)] =
        (T_dense_local[(14)] + (A_shared_local[(2)] * B_shared_local[(3)]));
    T_dense_local[(18)] =
        (T_dense_local[(18)] + (A_shared_local[(2)] * B_shared_local[(4)]));
    T_dense_local[(22)] =
        (T_dense_local[(22)] + (A_shared_local[(2)] * B_shared_local[(5)]));
    T_dense_local[(26)] =
        (T_dense_local[(26)] + (A_shared_local[(2)] * B_shared_local[(6)]));
    T_dense_local[(30)] =
        (T_dense_local[(30)] + (A_shared_local[(2)] * B_shared_local[(7)]));
    T_dense_local[(3)] =
        (T_dense_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    T_dense_local[(7)] =
        (T_dense_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    T_dense_local[(11)] =
        (T_dense_local[(11)] + (A_shared_local[(3)] * B_shared_local[(2)]));
    T_dense_local[(15)] =
        (T_dense_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
    T_dense_local[(19)] =
        (T_dense_local[(19)] + (A_shared_local[(3)] * B_shared_local[(4)]));
    T_dense_local[(23)] =
        (T_dense_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
    T_dense_local[(27)] =
        (T_dense_local[(27)] + (A_shared_local[(3)] * B_shared_local[(6)]));
    T_dense_local[(31)] =
        (T_dense_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
    A_shared_local[(0)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 3))];
    A_shared_local[(1)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 131))];
    A_shared_local[(2)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 259))];
    A_shared_local[(3)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 387))];
    B_shared_local[(0)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 3))];
    B_shared_local[(1)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 131))];
    B_shared_local[(2)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 259))];
    B_shared_local[(3)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 387))];
    B_shared_local[(4)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 515))];
    B_shared_local[(5)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 643))];
    B_shared_local[(6)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 771))];
    B_shared_local[(7)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 899))];
    T_dense_local[(0)] =
        (T_dense_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    T_dense_local[(4)] =
        (T_dense_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    T_dense_local[(8)] =
        (T_dense_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
    T_dense_local[(12)] =
        (T_dense_local[(12)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    T_dense_local[(16)] =
        (T_dense_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
    T_dense_local[(20)] =
        (T_dense_local[(20)] + (A_shared_local[(0)] * B_shared_local[(5)]));
    T_dense_local[(24)] =
        (T_dense_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
    T_dense_local[(28)] =
        (T_dense_local[(28)] + (A_shared_local[(0)] * B_shared_local[(7)]));
    T_dense_local[(1)] =
        (T_dense_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
    T_dense_local[(5)] =
        (T_dense_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
    T_dense_local[(9)] =
        (T_dense_local[(9)] + (A_shared_local[(1)] * B_shared_local[(2)]));
    T_dense_local[(13)] =
        (T_dense_local[(13)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    T_dense_local[(17)] =
        (T_dense_local[(17)] + (A_shared_local[(1)] * B_shared_local[(4)]));
    T_dense_local[(21)] =
        (T_dense_local[(21)] + (A_shared_local[(1)] * B_shared_local[(5)]));
    T_dense_local[(25)] =
        (T_dense_local[(25)] + (A_shared_local[(1)] * B_shared_local[(6)]));
    T_dense_local[(29)] =
        (T_dense_local[(29)] + (A_shared_local[(1)] * B_shared_local[(7)]));
    T_dense_local[(2)] =
        (T_dense_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
    T_dense_local[(6)] =
        (T_dense_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
    T_dense_local[(10)] =
        (T_dense_local[(10)] + (A_shared_local[(2)] * B_shared_local[(2)]));
    T_dense_local[(14)] =
        (T_dense_local[(14)] + (A_shared_local[(2)] * B_shared_local[(3)]));
    T_dense_local[(18)] =
        (T_dense_local[(18)] + (A_shared_local[(2)] * B_shared_local[(4)]));
    T_dense_local[(22)] =
        (T_dense_local[(22)] + (A_shared_local[(2)] * B_shared_local[(5)]));
    T_dense_local[(26)] =
        (T_dense_local[(26)] + (A_shared_local[(2)] * B_shared_local[(6)]));
    T_dense_local[(30)] =
        (T_dense_local[(30)] + (A_shared_local[(2)] * B_shared_local[(7)]));
    T_dense_local[(3)] =
        (T_dense_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    T_dense_local[(7)] =
        (T_dense_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    T_dense_local[(11)] =
        (T_dense_local[(11)] + (A_shared_local[(3)] * B_shared_local[(2)]));
    T_dense_local[(15)] =
        (T_dense_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
    T_dense_local[(19)] =
        (T_dense_local[(19)] + (A_shared_local[(3)] * B_shared_local[(4)]));
    T_dense_local[(23)] =
        (T_dense_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
    T_dense_local[(27)] =
        (T_dense_local[(27)] + (A_shared_local[(3)] * B_shared_local[(6)]));
    T_dense_local[(31)] =
        (T_dense_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
    A_shared_local[(0)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 4))];
    A_shared_local[(1)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 132))];
    A_shared_local[(2)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 260))];
    A_shared_local[(3)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 388))];
    B_shared_local[(0)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 4))];
    B_shared_local[(1)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 132))];
    B_shared_local[(2)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 260))];
    B_shared_local[(3)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 388))];
    B_shared_local[(4)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 516))];
    B_shared_local[(5)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 644))];
    B_shared_local[(6)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 772))];
    B_shared_local[(7)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 900))];
    T_dense_local[(0)] =
        (T_dense_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    T_dense_local[(4)] =
        (T_dense_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    T_dense_local[(8)] =
        (T_dense_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
    T_dense_local[(12)] =
        (T_dense_local[(12)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    T_dense_local[(16)] =
        (T_dense_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
    T_dense_local[(20)] =
        (T_dense_local[(20)] + (A_shared_local[(0)] * B_shared_local[(5)]));
    T_dense_local[(24)] =
        (T_dense_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
    T_dense_local[(28)] =
        (T_dense_local[(28)] + (A_shared_local[(0)] * B_shared_local[(7)]));
    T_dense_local[(1)] =
        (T_dense_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
    T_dense_local[(5)] =
        (T_dense_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
    T_dense_local[(9)] =
        (T_dense_local[(9)] + (A_shared_local[(1)] * B_shared_local[(2)]));
    T_dense_local[(13)] =
        (T_dense_local[(13)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    T_dense_local[(17)] =
        (T_dense_local[(17)] + (A_shared_local[(1)] * B_shared_local[(4)]));
    T_dense_local[(21)] =
        (T_dense_local[(21)] + (A_shared_local[(1)] * B_shared_local[(5)]));
    T_dense_local[(25)] =
        (T_dense_local[(25)] + (A_shared_local[(1)] * B_shared_local[(6)]));
    T_dense_local[(29)] =
        (T_dense_local[(29)] + (A_shared_local[(1)] * B_shared_local[(7)]));
    T_dense_local[(2)] =
        (T_dense_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
    T_dense_local[(6)] =
        (T_dense_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
    T_dense_local[(10)] =
        (T_dense_local[(10)] + (A_shared_local[(2)] * B_shared_local[(2)]));
    T_dense_local[(14)] =
        (T_dense_local[(14)] + (A_shared_local[(2)] * B_shared_local[(3)]));
    T_dense_local[(18)] =
        (T_dense_local[(18)] + (A_shared_local[(2)] * B_shared_local[(4)]));
    T_dense_local[(22)] =
        (T_dense_local[(22)] + (A_shared_local[(2)] * B_shared_local[(5)]));
    T_dense_local[(26)] =
        (T_dense_local[(26)] + (A_shared_local[(2)] * B_shared_local[(6)]));
    T_dense_local[(30)] =
        (T_dense_local[(30)] + (A_shared_local[(2)] * B_shared_local[(7)]));
    T_dense_local[(3)] =
        (T_dense_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    T_dense_local[(7)] =
        (T_dense_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    T_dense_local[(11)] =
        (T_dense_local[(11)] + (A_shared_local[(3)] * B_shared_local[(2)]));
    T_dense_local[(15)] =
        (T_dense_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
    T_dense_local[(19)] =
        (T_dense_local[(19)] + (A_shared_local[(3)] * B_shared_local[(4)]));
    T_dense_local[(23)] =
        (T_dense_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
    T_dense_local[(27)] =
        (T_dense_local[(27)] + (A_shared_local[(3)] * B_shared_local[(6)]));
    T_dense_local[(31)] =
        (T_dense_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
    A_shared_local[(0)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 5))];
    A_shared_local[(1)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 133))];
    A_shared_local[(2)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 261))];
    A_shared_local[(3)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 389))];
    B_shared_local[(0)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 5))];
    B_shared_local[(1)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 133))];
    B_shared_local[(2)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 261))];
    B_shared_local[(3)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 389))];
    B_shared_local[(4)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 517))];
    B_shared_local[(5)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 645))];
    B_shared_local[(6)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 773))];
    B_shared_local[(7)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 901))];
    T_dense_local[(0)] =
        (T_dense_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    T_dense_local[(4)] =
        (T_dense_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    T_dense_local[(8)] =
        (T_dense_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
    T_dense_local[(12)] =
        (T_dense_local[(12)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    T_dense_local[(16)] =
        (T_dense_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
    T_dense_local[(20)] =
        (T_dense_local[(20)] + (A_shared_local[(0)] * B_shared_local[(5)]));
    T_dense_local[(24)] =
        (T_dense_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
    T_dense_local[(28)] =
        (T_dense_local[(28)] + (A_shared_local[(0)] * B_shared_local[(7)]));
    T_dense_local[(1)] =
        (T_dense_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
    T_dense_local[(5)] =
        (T_dense_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
    T_dense_local[(9)] =
        (T_dense_local[(9)] + (A_shared_local[(1)] * B_shared_local[(2)]));
    T_dense_local[(13)] =
        (T_dense_local[(13)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    T_dense_local[(17)] =
        (T_dense_local[(17)] + (A_shared_local[(1)] * B_shared_local[(4)]));
    T_dense_local[(21)] =
        (T_dense_local[(21)] + (A_shared_local[(1)] * B_shared_local[(5)]));
    T_dense_local[(25)] =
        (T_dense_local[(25)] + (A_shared_local[(1)] * B_shared_local[(6)]));
    T_dense_local[(29)] =
        (T_dense_local[(29)] + (A_shared_local[(1)] * B_shared_local[(7)]));
    T_dense_local[(2)] =
        (T_dense_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
    T_dense_local[(6)] =
        (T_dense_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
    T_dense_local[(10)] =
        (T_dense_local[(10)] + (A_shared_local[(2)] * B_shared_local[(2)]));
    T_dense_local[(14)] =
        (T_dense_local[(14)] + (A_shared_local[(2)] * B_shared_local[(3)]));
    T_dense_local[(18)] =
        (T_dense_local[(18)] + (A_shared_local[(2)] * B_shared_local[(4)]));
    T_dense_local[(22)] =
        (T_dense_local[(22)] + (A_shared_local[(2)] * B_shared_local[(5)]));
    T_dense_local[(26)] =
        (T_dense_local[(26)] + (A_shared_local[(2)] * B_shared_local[(6)]));
    T_dense_local[(30)] =
        (T_dense_local[(30)] + (A_shared_local[(2)] * B_shared_local[(7)]));
    T_dense_local[(3)] =
        (T_dense_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    T_dense_local[(7)] =
        (T_dense_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    T_dense_local[(11)] =
        (T_dense_local[(11)] + (A_shared_local[(3)] * B_shared_local[(2)]));
    T_dense_local[(15)] =
        (T_dense_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
    T_dense_local[(19)] =
        (T_dense_local[(19)] + (A_shared_local[(3)] * B_shared_local[(4)]));
    T_dense_local[(23)] =
        (T_dense_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
    T_dense_local[(27)] =
        (T_dense_local[(27)] + (A_shared_local[(3)] * B_shared_local[(6)]));
    T_dense_local[(31)] =
        (T_dense_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
    A_shared_local[(0)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 6))];
    A_shared_local[(1)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 134))];
    A_shared_local[(2)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 262))];
    A_shared_local[(3)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 390))];
    B_shared_local[(0)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 6))];
    B_shared_local[(1)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 134))];
    B_shared_local[(2)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 262))];
    B_shared_local[(3)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 390))];
    B_shared_local[(4)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 518))];
    B_shared_local[(5)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 646))];
    B_shared_local[(6)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 774))];
    B_shared_local[(7)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 902))];
    T_dense_local[(0)] =
        (T_dense_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    T_dense_local[(4)] =
        (T_dense_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    T_dense_local[(8)] =
        (T_dense_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
    T_dense_local[(12)] =
        (T_dense_local[(12)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    T_dense_local[(16)] =
        (T_dense_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
    T_dense_local[(20)] =
        (T_dense_local[(20)] + (A_shared_local[(0)] * B_shared_local[(5)]));
    T_dense_local[(24)] =
        (T_dense_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
    T_dense_local[(28)] =
        (T_dense_local[(28)] + (A_shared_local[(0)] * B_shared_local[(7)]));
    T_dense_local[(1)] =
        (T_dense_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
    T_dense_local[(5)] =
        (T_dense_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
    T_dense_local[(9)] =
        (T_dense_local[(9)] + (A_shared_local[(1)] * B_shared_local[(2)]));
    T_dense_local[(13)] =
        (T_dense_local[(13)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    T_dense_local[(17)] =
        (T_dense_local[(17)] + (A_shared_local[(1)] * B_shared_local[(4)]));
    T_dense_local[(21)] =
        (T_dense_local[(21)] + (A_shared_local[(1)] * B_shared_local[(5)]));
    T_dense_local[(25)] =
        (T_dense_local[(25)] + (A_shared_local[(1)] * B_shared_local[(6)]));
    T_dense_local[(29)] =
        (T_dense_local[(29)] + (A_shared_local[(1)] * B_shared_local[(7)]));
    T_dense_local[(2)] =
        (T_dense_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
    T_dense_local[(6)] =
        (T_dense_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
    T_dense_local[(10)] =
        (T_dense_local[(10)] + (A_shared_local[(2)] * B_shared_local[(2)]));
    T_dense_local[(14)] =
        (T_dense_local[(14)] + (A_shared_local[(2)] * B_shared_local[(3)]));
    T_dense_local[(18)] =
        (T_dense_local[(18)] + (A_shared_local[(2)] * B_shared_local[(4)]));
    T_dense_local[(22)] =
        (T_dense_local[(22)] + (A_shared_local[(2)] * B_shared_local[(5)]));
    T_dense_local[(26)] =
        (T_dense_local[(26)] + (A_shared_local[(2)] * B_shared_local[(6)]));
    T_dense_local[(30)] =
        (T_dense_local[(30)] + (A_shared_local[(2)] * B_shared_local[(7)]));
    T_dense_local[(3)] =
        (T_dense_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    T_dense_local[(7)] =
        (T_dense_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    T_dense_local[(11)] =
        (T_dense_local[(11)] + (A_shared_local[(3)] * B_shared_local[(2)]));
    T_dense_local[(15)] =
        (T_dense_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
    T_dense_local[(19)] =
        (T_dense_local[(19)] + (A_shared_local[(3)] * B_shared_local[(4)]));
    T_dense_local[(23)] =
        (T_dense_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
    T_dense_local[(27)] =
        (T_dense_local[(27)] + (A_shared_local[(3)] * B_shared_local[(6)]));
    T_dense_local[(31)] =
        (T_dense_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
    A_shared_local[(0)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 7))];
    A_shared_local[(1)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 135))];
    A_shared_local[(2)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 263))];
    A_shared_local[(3)] = A_shared[(
        ((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 391))];
    B_shared_local[(0)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 7))];
    B_shared_local[(1)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 135))];
    B_shared_local[(2)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 263))];
    B_shared_local[(3)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 391))];
    B_shared_local[(4)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 519))];
    B_shared_local[(5)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 647))];
    B_shared_local[(6)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 775))];
    B_shared_local[(7)] = B_shared[(
        ((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 8)) + 903))];
    T_dense_local[(0)] =
        (T_dense_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    T_dense_local[(4)] =
        (T_dense_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    T_dense_local[(8)] =
        (T_dense_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
    T_dense_local[(12)] =
        (T_dense_local[(12)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    T_dense_local[(16)] =
        (T_dense_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
    T_dense_local[(20)] =
        (T_dense_local[(20)] + (A_shared_local[(0)] * B_shared_local[(5)]));
    T_dense_local[(24)] =
        (T_dense_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
    T_dense_local[(28)] =
        (T_dense_local[(28)] + (A_shared_local[(0)] * B_shared_local[(7)]));
    T_dense_local[(1)] =
        (T_dense_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
    T_dense_local[(5)] =
        (T_dense_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
    T_dense_local[(9)] =
        (T_dense_local[(9)] + (A_shared_local[(1)] * B_shared_local[(2)]));
    T_dense_local[(13)] =
        (T_dense_local[(13)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    T_dense_local[(17)] =
        (T_dense_local[(17)] + (A_shared_local[(1)] * B_shared_local[(4)]));
    T_dense_local[(21)] =
        (T_dense_local[(21)] + (A_shared_local[(1)] * B_shared_local[(5)]));
    T_dense_local[(25)] =
        (T_dense_local[(25)] + (A_shared_local[(1)] * B_shared_local[(6)]));
    T_dense_local[(29)] =
        (T_dense_local[(29)] + (A_shared_local[(1)] * B_shared_local[(7)]));
    T_dense_local[(2)] =
        (T_dense_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
    T_dense_local[(6)] =
        (T_dense_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
    T_dense_local[(10)] =
        (T_dense_local[(10)] + (A_shared_local[(2)] * B_shared_local[(2)]));
    T_dense_local[(14)] =
        (T_dense_local[(14)] + (A_shared_local[(2)] * B_shared_local[(3)]));
    T_dense_local[(18)] =
        (T_dense_local[(18)] + (A_shared_local[(2)] * B_shared_local[(4)]));
    T_dense_local[(22)] =
        (T_dense_local[(22)] + (A_shared_local[(2)] * B_shared_local[(5)]));
    T_dense_local[(26)] =
        (T_dense_local[(26)] + (A_shared_local[(2)] * B_shared_local[(6)]));
    T_dense_local[(30)] =
        (T_dense_local[(30)] + (A_shared_local[(2)] * B_shared_local[(7)]));
    T_dense_local[(3)] =
        (T_dense_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    T_dense_local[(7)] =
        (T_dense_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    T_dense_local[(11)] =
        (T_dense_local[(11)] + (A_shared_local[(3)] * B_shared_local[(2)]));
    T_dense_local[(15)] =
        (T_dense_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
    T_dense_local[(19)] =
        (T_dense_local[(19)] + (A_shared_local[(3)] * B_shared_local[(4)]));
    T_dense_local[(23)] =
        (T_dense_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
    T_dense_local[(27)] =
        (T_dense_local[(27)] + (A_shared_local[(3)] * B_shared_local[(6)]));
    T_dense_local[(31)] =
        (T_dense_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
  }
  __syncthreads();
  A_shared_local1[(0)] = A_shared[(((((int)threadIdx.x) * 8) + 512))];
  A_shared_local1[(1)] = A_shared[(((((int)threadIdx.x) * 8) + 640))];
  A_shared_local1[(2)] = A_shared[(((((int)threadIdx.x) * 8) + 768))];
  A_shared_local1[(3)] = A_shared[(((((int)threadIdx.x) * 8) + 896))];
  B_shared_local1[(0)] = B_shared[(((((int)threadIdx.y) * 8) + 1024))];
  B_shared_local1[(1)] = B_shared[(((((int)threadIdx.y) * 8) + 1152))];
  B_shared_local1[(2)] = B_shared[(((((int)threadIdx.y) * 8) + 1280))];
  B_shared_local1[(3)] = B_shared[(((((int)threadIdx.y) * 8) + 1408))];
  B_shared_local1[(4)] = B_shared[(((((int)threadIdx.y) * 8) + 1536))];
  B_shared_local1[(5)] = B_shared[(((((int)threadIdx.y) * 8) + 1664))];
  B_shared_local1[(6)] = B_shared[(((((int)threadIdx.y) * 8) + 1792))];
  B_shared_local1[(7)] = B_shared[(((((int)threadIdx.y) * 8) + 1920))];
  T_dense_local[(0)] =
      (T_dense_local[(0)] + (A_shared_local1[(0)] * B_shared_local1[(0)]));
  T_dense_local[(4)] =
      (T_dense_local[(4)] + (A_shared_local1[(0)] * B_shared_local1[(1)]));
  T_dense_local[(8)] =
      (T_dense_local[(8)] + (A_shared_local1[(0)] * B_shared_local1[(2)]));
  T_dense_local[(12)] =
      (T_dense_local[(12)] + (A_shared_local1[(0)] * B_shared_local1[(3)]));
  T_dense_local[(16)] =
      (T_dense_local[(16)] + (A_shared_local1[(0)] * B_shared_local1[(4)]));
  T_dense_local[(20)] =
      (T_dense_local[(20)] + (A_shared_local1[(0)] * B_shared_local1[(5)]));
  T_dense_local[(24)] =
      (T_dense_local[(24)] + (A_shared_local1[(0)] * B_shared_local1[(6)]));
  T_dense_local[(28)] =
      (T_dense_local[(28)] + (A_shared_local1[(0)] * B_shared_local1[(7)]));
  T_dense_local[(1)] =
      (T_dense_local[(1)] + (A_shared_local1[(1)] * B_shared_local1[(0)]));
  T_dense_local[(5)] =
      (T_dense_local[(5)] + (A_shared_local1[(1)] * B_shared_local1[(1)]));
  T_dense_local[(9)] =
      (T_dense_local[(9)] + (A_shared_local1[(1)] * B_shared_local1[(2)]));
  T_dense_local[(13)] =
      (T_dense_local[(13)] + (A_shared_local1[(1)] * B_shared_local1[(3)]));
  T_dense_local[(17)] =
      (T_dense_local[(17)] + (A_shared_local1[(1)] * B_shared_local1[(4)]));
  T_dense_local[(21)] =
      (T_dense_local[(21)] + (A_shared_local1[(1)] * B_shared_local1[(5)]));
  T_dense_local[(25)] =
      (T_dense_local[(25)] + (A_shared_local1[(1)] * B_shared_local1[(6)]));
  T_dense_local[(29)] =
      (T_dense_local[(29)] + (A_shared_local1[(1)] * B_shared_local1[(7)]));
  T_dense_local[(2)] =
      (T_dense_local[(2)] + (A_shared_local1[(2)] * B_shared_local1[(0)]));
  T_dense_local[(6)] =
      (T_dense_local[(6)] + (A_shared_local1[(2)] * B_shared_local1[(1)]));
  T_dense_local[(10)] =
      (T_dense_local[(10)] + (A_shared_local1[(2)] * B_shared_local1[(2)]));
  T_dense_local[(14)] =
      (T_dense_local[(14)] + (A_shared_local1[(2)] * B_shared_local1[(3)]));
  T_dense_local[(18)] =
      (T_dense_local[(18)] + (A_shared_local1[(2)] * B_shared_local1[(4)]));
  T_dense_local[(22)] =
      (T_dense_local[(22)] + (A_shared_local1[(2)] * B_shared_local1[(5)]));
  T_dense_local[(26)] =
      (T_dense_local[(26)] + (A_shared_local1[(2)] * B_shared_local1[(6)]));
  T_dense_local[(30)] =
      (T_dense_local[(30)] + (A_shared_local1[(2)] * B_shared_local1[(7)]));
  T_dense_local[(3)] =
      (T_dense_local[(3)] + (A_shared_local1[(3)] * B_shared_local1[(0)]));
  T_dense_local[(7)] =
      (T_dense_local[(7)] + (A_shared_local1[(3)] * B_shared_local1[(1)]));
  T_dense_local[(11)] =
      (T_dense_local[(11)] + (A_shared_local1[(3)] * B_shared_local1[(2)]));
  T_dense_local[(15)] =
      (T_dense_local[(15)] + (A_shared_local1[(3)] * B_shared_local1[(3)]));
  T_dense_local[(19)] =
      (T_dense_local[(19)] + (A_shared_local1[(3)] * B_shared_local1[(4)]));
  T_dense_local[(23)] =
      (T_dense_local[(23)] + (A_shared_local1[(3)] * B_shared_local1[(5)]));
  T_dense_local[(27)] =
      (T_dense_local[(27)] + (A_shared_local1[(3)] * B_shared_local1[(6)]));
  T_dense_local[(31)] =
      (T_dense_local[(31)] + (A_shared_local1[(3)] * B_shared_local1[(7)]));
  A_shared_local1[(0)] = A_shared[(((((int)threadIdx.x) * 8) + 513))];
  A_shared_local1[(1)] = A_shared[(((((int)threadIdx.x) * 8) + 641))];
  A_shared_local1[(2)] = A_shared[(((((int)threadIdx.x) * 8) + 769))];
  A_shared_local1[(3)] = A_shared[(((((int)threadIdx.x) * 8) + 897))];
  B_shared_local1[(0)] = B_shared[(((((int)threadIdx.y) * 8) + 1025))];
  B_shared_local1[(1)] = B_shared[(((((int)threadIdx.y) * 8) + 1153))];
  B_shared_local1[(2)] = B_shared[(((((int)threadIdx.y) * 8) + 1281))];
  B_shared_local1[(3)] = B_shared[(((((int)threadIdx.y) * 8) + 1409))];
  B_shared_local1[(4)] = B_shared[(((((int)threadIdx.y) * 8) + 1537))];
  B_shared_local1[(5)] = B_shared[(((((int)threadIdx.y) * 8) + 1665))];
  B_shared_local1[(6)] = B_shared[(((((int)threadIdx.y) * 8) + 1793))];
  B_shared_local1[(7)] = B_shared[(((((int)threadIdx.y) * 8) + 1921))];
  T_dense_local[(0)] =
      (T_dense_local[(0)] + (A_shared_local1[(0)] * B_shared_local1[(0)]));
  T_dense_local[(4)] =
      (T_dense_local[(4)] + (A_shared_local1[(0)] * B_shared_local1[(1)]));
  T_dense_local[(8)] =
      (T_dense_local[(8)] + (A_shared_local1[(0)] * B_shared_local1[(2)]));
  T_dense_local[(12)] =
      (T_dense_local[(12)] + (A_shared_local1[(0)] * B_shared_local1[(3)]));
  T_dense_local[(16)] =
      (T_dense_local[(16)] + (A_shared_local1[(0)] * B_shared_local1[(4)]));
  T_dense_local[(20)] =
      (T_dense_local[(20)] + (A_shared_local1[(0)] * B_shared_local1[(5)]));
  T_dense_local[(24)] =
      (T_dense_local[(24)] + (A_shared_local1[(0)] * B_shared_local1[(6)]));
  T_dense_local[(28)] =
      (T_dense_local[(28)] + (A_shared_local1[(0)] * B_shared_local1[(7)]));
  T_dense_local[(1)] =
      (T_dense_local[(1)] + (A_shared_local1[(1)] * B_shared_local1[(0)]));
  T_dense_local[(5)] =
      (T_dense_local[(5)] + (A_shared_local1[(1)] * B_shared_local1[(1)]));
  T_dense_local[(9)] =
      (T_dense_local[(9)] + (A_shared_local1[(1)] * B_shared_local1[(2)]));
  T_dense_local[(13)] =
      (T_dense_local[(13)] + (A_shared_local1[(1)] * B_shared_local1[(3)]));
  T_dense_local[(17)] =
      (T_dense_local[(17)] + (A_shared_local1[(1)] * B_shared_local1[(4)]));
  T_dense_local[(21)] =
      (T_dense_local[(21)] + (A_shared_local1[(1)] * B_shared_local1[(5)]));
  T_dense_local[(25)] =
      (T_dense_local[(25)] + (A_shared_local1[(1)] * B_shared_local1[(6)]));
  T_dense_local[(29)] =
      (T_dense_local[(29)] + (A_shared_local1[(1)] * B_shared_local1[(7)]));
  T_dense_local[(2)] =
      (T_dense_local[(2)] + (A_shared_local1[(2)] * B_shared_local1[(0)]));
  T_dense_local[(6)] =
      (T_dense_local[(6)] + (A_shared_local1[(2)] * B_shared_local1[(1)]));
  T_dense_local[(10)] =
      (T_dense_local[(10)] + (A_shared_local1[(2)] * B_shared_local1[(2)]));
  T_dense_local[(14)] =
      (T_dense_local[(14)] + (A_shared_local1[(2)] * B_shared_local1[(3)]));
  T_dense_local[(18)] =
      (T_dense_local[(18)] + (A_shared_local1[(2)] * B_shared_local1[(4)]));
  T_dense_local[(22)] =
      (T_dense_local[(22)] + (A_shared_local1[(2)] * B_shared_local1[(5)]));
  T_dense_local[(26)] =
      (T_dense_local[(26)] + (A_shared_local1[(2)] * B_shared_local1[(6)]));
  T_dense_local[(30)] =
      (T_dense_local[(30)] + (A_shared_local1[(2)] * B_shared_local1[(7)]));
  T_dense_local[(3)] =
      (T_dense_local[(3)] + (A_shared_local1[(3)] * B_shared_local1[(0)]));
  T_dense_local[(7)] =
      (T_dense_local[(7)] + (A_shared_local1[(3)] * B_shared_local1[(1)]));
  T_dense_local[(11)] =
      (T_dense_local[(11)] + (A_shared_local1[(3)] * B_shared_local1[(2)]));
  T_dense_local[(15)] =
      (T_dense_local[(15)] + (A_shared_local1[(3)] * B_shared_local1[(3)]));
  T_dense_local[(19)] =
      (T_dense_local[(19)] + (A_shared_local1[(3)] * B_shared_local1[(4)]));
  T_dense_local[(23)] =
      (T_dense_local[(23)] + (A_shared_local1[(3)] * B_shared_local1[(5)]));
  T_dense_local[(27)] =
      (T_dense_local[(27)] + (A_shared_local1[(3)] * B_shared_local1[(6)]));
  T_dense_local[(31)] =
      (T_dense_local[(31)] + (A_shared_local1[(3)] * B_shared_local1[(7)]));
  A_shared_local1[(0)] = A_shared[(((((int)threadIdx.x) * 8) + 514))];
  A_shared_local1[(1)] = A_shared[(((((int)threadIdx.x) * 8) + 642))];
  A_shared_local1[(2)] = A_shared[(((((int)threadIdx.x) * 8) + 770))];
  A_shared_local1[(3)] = A_shared[(((((int)threadIdx.x) * 8) + 898))];
  B_shared_local1[(0)] = B_shared[(((((int)threadIdx.y) * 8) + 1026))];
  B_shared_local1[(1)] = B_shared[(((((int)threadIdx.y) * 8) + 1154))];
  B_shared_local1[(2)] = B_shared[(((((int)threadIdx.y) * 8) + 1282))];
  B_shared_local1[(3)] = B_shared[(((((int)threadIdx.y) * 8) + 1410))];
  B_shared_local1[(4)] = B_shared[(((((int)threadIdx.y) * 8) + 1538))];
  B_shared_local1[(5)] = B_shared[(((((int)threadIdx.y) * 8) + 1666))];
  B_shared_local1[(6)] = B_shared[(((((int)threadIdx.y) * 8) + 1794))];
  B_shared_local1[(7)] = B_shared[(((((int)threadIdx.y) * 8) + 1922))];
  T_dense_local[(0)] =
      (T_dense_local[(0)] + (A_shared_local1[(0)] * B_shared_local1[(0)]));
  T_dense_local[(4)] =
      (T_dense_local[(4)] + (A_shared_local1[(0)] * B_shared_local1[(1)]));
  T_dense_local[(8)] =
      (T_dense_local[(8)] + (A_shared_local1[(0)] * B_shared_local1[(2)]));
  T_dense_local[(12)] =
      (T_dense_local[(12)] + (A_shared_local1[(0)] * B_shared_local1[(3)]));
  T_dense_local[(16)] =
      (T_dense_local[(16)] + (A_shared_local1[(0)] * B_shared_local1[(4)]));
  T_dense_local[(20)] =
      (T_dense_local[(20)] + (A_shared_local1[(0)] * B_shared_local1[(5)]));
  T_dense_local[(24)] =
      (T_dense_local[(24)] + (A_shared_local1[(0)] * B_shared_local1[(6)]));
  T_dense_local[(28)] =
      (T_dense_local[(28)] + (A_shared_local1[(0)] * B_shared_local1[(7)]));
  T_dense_local[(1)] =
      (T_dense_local[(1)] + (A_shared_local1[(1)] * B_shared_local1[(0)]));
  T_dense_local[(5)] =
      (T_dense_local[(5)] + (A_shared_local1[(1)] * B_shared_local1[(1)]));
  T_dense_local[(9)] =
      (T_dense_local[(9)] + (A_shared_local1[(1)] * B_shared_local1[(2)]));
  T_dense_local[(13)] =
      (T_dense_local[(13)] + (A_shared_local1[(1)] * B_shared_local1[(3)]));
  T_dense_local[(17)] =
      (T_dense_local[(17)] + (A_shared_local1[(1)] * B_shared_local1[(4)]));
  T_dense_local[(21)] =
      (T_dense_local[(21)] + (A_shared_local1[(1)] * B_shared_local1[(5)]));
  T_dense_local[(25)] =
      (T_dense_local[(25)] + (A_shared_local1[(1)] * B_shared_local1[(6)]));
  T_dense_local[(29)] =
      (T_dense_local[(29)] + (A_shared_local1[(1)] * B_shared_local1[(7)]));
  T_dense_local[(2)] =
      (T_dense_local[(2)] + (A_shared_local1[(2)] * B_shared_local1[(0)]));
  T_dense_local[(6)] =
      (T_dense_local[(6)] + (A_shared_local1[(2)] * B_shared_local1[(1)]));
  T_dense_local[(10)] =
      (T_dense_local[(10)] + (A_shared_local1[(2)] * B_shared_local1[(2)]));
  T_dense_local[(14)] =
      (T_dense_local[(14)] + (A_shared_local1[(2)] * B_shared_local1[(3)]));
  T_dense_local[(18)] =
      (T_dense_local[(18)] + (A_shared_local1[(2)] * B_shared_local1[(4)]));
  T_dense_local[(22)] =
      (T_dense_local[(22)] + (A_shared_local1[(2)] * B_shared_local1[(5)]));
  T_dense_local[(26)] =
      (T_dense_local[(26)] + (A_shared_local1[(2)] * B_shared_local1[(6)]));
  T_dense_local[(30)] =
      (T_dense_local[(30)] + (A_shared_local1[(2)] * B_shared_local1[(7)]));
  T_dense_local[(3)] =
      (T_dense_local[(3)] + (A_shared_local1[(3)] * B_shared_local1[(0)]));
  T_dense_local[(7)] =
      (T_dense_local[(7)] + (A_shared_local1[(3)] * B_shared_local1[(1)]));
  T_dense_local[(11)] =
      (T_dense_local[(11)] + (A_shared_local1[(3)] * B_shared_local1[(2)]));
  T_dense_local[(15)] =
      (T_dense_local[(15)] + (A_shared_local1[(3)] * B_shared_local1[(3)]));
  T_dense_local[(19)] =
      (T_dense_local[(19)] + (A_shared_local1[(3)] * B_shared_local1[(4)]));
  T_dense_local[(23)] =
      (T_dense_local[(23)] + (A_shared_local1[(3)] * B_shared_local1[(5)]));
  T_dense_local[(27)] =
      (T_dense_local[(27)] + (A_shared_local1[(3)] * B_shared_local1[(6)]));
  T_dense_local[(31)] =
      (T_dense_local[(31)] + (A_shared_local1[(3)] * B_shared_local1[(7)]));
  A_shared_local1[(0)] = A_shared[(((((int)threadIdx.x) * 8) + 515))];
  A_shared_local1[(1)] = A_shared[(((((int)threadIdx.x) * 8) + 643))];
  A_shared_local1[(2)] = A_shared[(((((int)threadIdx.x) * 8) + 771))];
  A_shared_local1[(3)] = A_shared[(((((int)threadIdx.x) * 8) + 899))];
  B_shared_local1[(0)] = B_shared[(((((int)threadIdx.y) * 8) + 1027))];
  B_shared_local1[(1)] = B_shared[(((((int)threadIdx.y) * 8) + 1155))];
  B_shared_local1[(2)] = B_shared[(((((int)threadIdx.y) * 8) + 1283))];
  B_shared_local1[(3)] = B_shared[(((((int)threadIdx.y) * 8) + 1411))];
  B_shared_local1[(4)] = B_shared[(((((int)threadIdx.y) * 8) + 1539))];
  B_shared_local1[(5)] = B_shared[(((((int)threadIdx.y) * 8) + 1667))];
  B_shared_local1[(6)] = B_shared[(((((int)threadIdx.y) * 8) + 1795))];
  B_shared_local1[(7)] = B_shared[(((((int)threadIdx.y) * 8) + 1923))];
  T_dense_local[(0)] =
      (T_dense_local[(0)] + (A_shared_local1[(0)] * B_shared_local1[(0)]));
  T_dense_local[(4)] =
      (T_dense_local[(4)] + (A_shared_local1[(0)] * B_shared_local1[(1)]));
  T_dense_local[(8)] =
      (T_dense_local[(8)] + (A_shared_local1[(0)] * B_shared_local1[(2)]));
  T_dense_local[(12)] =
      (T_dense_local[(12)] + (A_shared_local1[(0)] * B_shared_local1[(3)]));
  T_dense_local[(16)] =
      (T_dense_local[(16)] + (A_shared_local1[(0)] * B_shared_local1[(4)]));
  T_dense_local[(20)] =
      (T_dense_local[(20)] + (A_shared_local1[(0)] * B_shared_local1[(5)]));
  T_dense_local[(24)] =
      (T_dense_local[(24)] + (A_shared_local1[(0)] * B_shared_local1[(6)]));
  T_dense_local[(28)] =
      (T_dense_local[(28)] + (A_shared_local1[(0)] * B_shared_local1[(7)]));
  T_dense_local[(1)] =
      (T_dense_local[(1)] + (A_shared_local1[(1)] * B_shared_local1[(0)]));
  T_dense_local[(5)] =
      (T_dense_local[(5)] + (A_shared_local1[(1)] * B_shared_local1[(1)]));
  T_dense_local[(9)] =
      (T_dense_local[(9)] + (A_shared_local1[(1)] * B_shared_local1[(2)]));
  T_dense_local[(13)] =
      (T_dense_local[(13)] + (A_shared_local1[(1)] * B_shared_local1[(3)]));
  T_dense_local[(17)] =
      (T_dense_local[(17)] + (A_shared_local1[(1)] * B_shared_local1[(4)]));
  T_dense_local[(21)] =
      (T_dense_local[(21)] + (A_shared_local1[(1)] * B_shared_local1[(5)]));
  T_dense_local[(25)] =
      (T_dense_local[(25)] + (A_shared_local1[(1)] * B_shared_local1[(6)]));
  T_dense_local[(29)] =
      (T_dense_local[(29)] + (A_shared_local1[(1)] * B_shared_local1[(7)]));
  T_dense_local[(2)] =
      (T_dense_local[(2)] + (A_shared_local1[(2)] * B_shared_local1[(0)]));
  T_dense_local[(6)] =
      (T_dense_local[(6)] + (A_shared_local1[(2)] * B_shared_local1[(1)]));
  T_dense_local[(10)] =
      (T_dense_local[(10)] + (A_shared_local1[(2)] * B_shared_local1[(2)]));
  T_dense_local[(14)] =
      (T_dense_local[(14)] + (A_shared_local1[(2)] * B_shared_local1[(3)]));
  T_dense_local[(18)] =
      (T_dense_local[(18)] + (A_shared_local1[(2)] * B_shared_local1[(4)]));
  T_dense_local[(22)] =
      (T_dense_local[(22)] + (A_shared_local1[(2)] * B_shared_local1[(5)]));
  T_dense_local[(26)] =
      (T_dense_local[(26)] + (A_shared_local1[(2)] * B_shared_local1[(6)]));
  T_dense_local[(30)] =
      (T_dense_local[(30)] + (A_shared_local1[(2)] * B_shared_local1[(7)]));
  T_dense_local[(3)] =
      (T_dense_local[(3)] + (A_shared_local1[(3)] * B_shared_local1[(0)]));
  T_dense_local[(7)] =
      (T_dense_local[(7)] + (A_shared_local1[(3)] * B_shared_local1[(1)]));
  T_dense_local[(11)] =
      (T_dense_local[(11)] + (A_shared_local1[(3)] * B_shared_local1[(2)]));
  T_dense_local[(15)] =
      (T_dense_local[(15)] + (A_shared_local1[(3)] * B_shared_local1[(3)]));
  T_dense_local[(19)] =
      (T_dense_local[(19)] + (A_shared_local1[(3)] * B_shared_local1[(4)]));
  T_dense_local[(23)] =
      (T_dense_local[(23)] + (A_shared_local1[(3)] * B_shared_local1[(5)]));
  T_dense_local[(27)] =
      (T_dense_local[(27)] + (A_shared_local1[(3)] * B_shared_local1[(6)]));
  T_dense_local[(31)] =
      (T_dense_local[(31)] + (A_shared_local1[(3)] * B_shared_local1[(7)]));
  A_shared_local1[(0)] = A_shared[(((((int)threadIdx.x) * 8) + 516))];
  A_shared_local1[(1)] = A_shared[(((((int)threadIdx.x) * 8) + 644))];
  A_shared_local1[(2)] = A_shared[(((((int)threadIdx.x) * 8) + 772))];
  A_shared_local1[(3)] = A_shared[(((((int)threadIdx.x) * 8) + 900))];
  B_shared_local1[(0)] = B_shared[(((((int)threadIdx.y) * 8) + 1028))];
  B_shared_local1[(1)] = B_shared[(((((int)threadIdx.y) * 8) + 1156))];
  B_shared_local1[(2)] = B_shared[(((((int)threadIdx.y) * 8) + 1284))];
  B_shared_local1[(3)] = B_shared[(((((int)threadIdx.y) * 8) + 1412))];
  B_shared_local1[(4)] = B_shared[(((((int)threadIdx.y) * 8) + 1540))];
  B_shared_local1[(5)] = B_shared[(((((int)threadIdx.y) * 8) + 1668))];
  B_shared_local1[(6)] = B_shared[(((((int)threadIdx.y) * 8) + 1796))];
  B_shared_local1[(7)] = B_shared[(((((int)threadIdx.y) * 8) + 1924))];
  T_dense_local[(0)] =
      (T_dense_local[(0)] + (A_shared_local1[(0)] * B_shared_local1[(0)]));
  T_dense_local[(4)] =
      (T_dense_local[(4)] + (A_shared_local1[(0)] * B_shared_local1[(1)]));
  T_dense_local[(8)] =
      (T_dense_local[(8)] + (A_shared_local1[(0)] * B_shared_local1[(2)]));
  T_dense_local[(12)] =
      (T_dense_local[(12)] + (A_shared_local1[(0)] * B_shared_local1[(3)]));
  T_dense_local[(16)] =
      (T_dense_local[(16)] + (A_shared_local1[(0)] * B_shared_local1[(4)]));
  T_dense_local[(20)] =
      (T_dense_local[(20)] + (A_shared_local1[(0)] * B_shared_local1[(5)]));
  T_dense_local[(24)] =
      (T_dense_local[(24)] + (A_shared_local1[(0)] * B_shared_local1[(6)]));
  T_dense_local[(28)] =
      (T_dense_local[(28)] + (A_shared_local1[(0)] * B_shared_local1[(7)]));
  T_dense_local[(1)] =
      (T_dense_local[(1)] + (A_shared_local1[(1)] * B_shared_local1[(0)]));
  T_dense_local[(5)] =
      (T_dense_local[(5)] + (A_shared_local1[(1)] * B_shared_local1[(1)]));
  T_dense_local[(9)] =
      (T_dense_local[(9)] + (A_shared_local1[(1)] * B_shared_local1[(2)]));
  T_dense_local[(13)] =
      (T_dense_local[(13)] + (A_shared_local1[(1)] * B_shared_local1[(3)]));
  T_dense_local[(17)] =
      (T_dense_local[(17)] + (A_shared_local1[(1)] * B_shared_local1[(4)]));
  T_dense_local[(21)] =
      (T_dense_local[(21)] + (A_shared_local1[(1)] * B_shared_local1[(5)]));
  T_dense_local[(25)] =
      (T_dense_local[(25)] + (A_shared_local1[(1)] * B_shared_local1[(6)]));
  T_dense_local[(29)] =
      (T_dense_local[(29)] + (A_shared_local1[(1)] * B_shared_local1[(7)]));
  T_dense_local[(2)] =
      (T_dense_local[(2)] + (A_shared_local1[(2)] * B_shared_local1[(0)]));
  T_dense_local[(6)] =
      (T_dense_local[(6)] + (A_shared_local1[(2)] * B_shared_local1[(1)]));
  T_dense_local[(10)] =
      (T_dense_local[(10)] + (A_shared_local1[(2)] * B_shared_local1[(2)]));
  T_dense_local[(14)] =
      (T_dense_local[(14)] + (A_shared_local1[(2)] * B_shared_local1[(3)]));
  T_dense_local[(18)] =
      (T_dense_local[(18)] + (A_shared_local1[(2)] * B_shared_local1[(4)]));
  T_dense_local[(22)] =
      (T_dense_local[(22)] + (A_shared_local1[(2)] * B_shared_local1[(5)]));
  T_dense_local[(26)] =
      (T_dense_local[(26)] + (A_shared_local1[(2)] * B_shared_local1[(6)]));
  T_dense_local[(30)] =
      (T_dense_local[(30)] + (A_shared_local1[(2)] * B_shared_local1[(7)]));
  T_dense_local[(3)] =
      (T_dense_local[(3)] + (A_shared_local1[(3)] * B_shared_local1[(0)]));
  T_dense_local[(7)] =
      (T_dense_local[(7)] + (A_shared_local1[(3)] * B_shared_local1[(1)]));
  T_dense_local[(11)] =
      (T_dense_local[(11)] + (A_shared_local1[(3)] * B_shared_local1[(2)]));
  T_dense_local[(15)] =
      (T_dense_local[(15)] + (A_shared_local1[(3)] * B_shared_local1[(3)]));
  T_dense_local[(19)] =
      (T_dense_local[(19)] + (A_shared_local1[(3)] * B_shared_local1[(4)]));
  T_dense_local[(23)] =
      (T_dense_local[(23)] + (A_shared_local1[(3)] * B_shared_local1[(5)]));
  T_dense_local[(27)] =
      (T_dense_local[(27)] + (A_shared_local1[(3)] * B_shared_local1[(6)]));
  T_dense_local[(31)] =
      (T_dense_local[(31)] + (A_shared_local1[(3)] * B_shared_local1[(7)]));
  A_shared_local1[(0)] = A_shared[(((((int)threadIdx.x) * 8) + 517))];
  A_shared_local1[(1)] = A_shared[(((((int)threadIdx.x) * 8) + 645))];
  A_shared_local1[(2)] = A_shared[(((((int)threadIdx.x) * 8) + 773))];
  A_shared_local1[(3)] = A_shared[(((((int)threadIdx.x) * 8) + 901))];
  B_shared_local1[(0)] = B_shared[(((((int)threadIdx.y) * 8) + 1029))];
  B_shared_local1[(1)] = B_shared[(((((int)threadIdx.y) * 8) + 1157))];
  B_shared_local1[(2)] = B_shared[(((((int)threadIdx.y) * 8) + 1285))];
  B_shared_local1[(3)] = B_shared[(((((int)threadIdx.y) * 8) + 1413))];
  B_shared_local1[(4)] = B_shared[(((((int)threadIdx.y) * 8) + 1541))];
  B_shared_local1[(5)] = B_shared[(((((int)threadIdx.y) * 8) + 1669))];
  B_shared_local1[(6)] = B_shared[(((((int)threadIdx.y) * 8) + 1797))];
  B_shared_local1[(7)] = B_shared[(((((int)threadIdx.y) * 8) + 1925))];
  T_dense_local[(0)] =
      (T_dense_local[(0)] + (A_shared_local1[(0)] * B_shared_local1[(0)]));
  T_dense_local[(4)] =
      (T_dense_local[(4)] + (A_shared_local1[(0)] * B_shared_local1[(1)]));
  T_dense_local[(8)] =
      (T_dense_local[(8)] + (A_shared_local1[(0)] * B_shared_local1[(2)]));
  T_dense_local[(12)] =
      (T_dense_local[(12)] + (A_shared_local1[(0)] * B_shared_local1[(3)]));
  T_dense_local[(16)] =
      (T_dense_local[(16)] + (A_shared_local1[(0)] * B_shared_local1[(4)]));
  T_dense_local[(20)] =
      (T_dense_local[(20)] + (A_shared_local1[(0)] * B_shared_local1[(5)]));
  T_dense_local[(24)] =
      (T_dense_local[(24)] + (A_shared_local1[(0)] * B_shared_local1[(6)]));
  T_dense_local[(28)] =
      (T_dense_local[(28)] + (A_shared_local1[(0)] * B_shared_local1[(7)]));
  T_dense_local[(1)] =
      (T_dense_local[(1)] + (A_shared_local1[(1)] * B_shared_local1[(0)]));
  T_dense_local[(5)] =
      (T_dense_local[(5)] + (A_shared_local1[(1)] * B_shared_local1[(1)]));
  T_dense_local[(9)] =
      (T_dense_local[(9)] + (A_shared_local1[(1)] * B_shared_local1[(2)]));
  T_dense_local[(13)] =
      (T_dense_local[(13)] + (A_shared_local1[(1)] * B_shared_local1[(3)]));
  T_dense_local[(17)] =
      (T_dense_local[(17)] + (A_shared_local1[(1)] * B_shared_local1[(4)]));
  T_dense_local[(21)] =
      (T_dense_local[(21)] + (A_shared_local1[(1)] * B_shared_local1[(5)]));
  T_dense_local[(25)] =
      (T_dense_local[(25)] + (A_shared_local1[(1)] * B_shared_local1[(6)]));
  T_dense_local[(29)] =
      (T_dense_local[(29)] + (A_shared_local1[(1)] * B_shared_local1[(7)]));
  T_dense_local[(2)] =
      (T_dense_local[(2)] + (A_shared_local1[(2)] * B_shared_local1[(0)]));
  T_dense_local[(6)] =
      (T_dense_local[(6)] + (A_shared_local1[(2)] * B_shared_local1[(1)]));
  T_dense_local[(10)] =
      (T_dense_local[(10)] + (A_shared_local1[(2)] * B_shared_local1[(2)]));
  T_dense_local[(14)] =
      (T_dense_local[(14)] + (A_shared_local1[(2)] * B_shared_local1[(3)]));
  T_dense_local[(18)] =
      (T_dense_local[(18)] + (A_shared_local1[(2)] * B_shared_local1[(4)]));
  T_dense_local[(22)] =
      (T_dense_local[(22)] + (A_shared_local1[(2)] * B_shared_local1[(5)]));
  T_dense_local[(26)] =
      (T_dense_local[(26)] + (A_shared_local1[(2)] * B_shared_local1[(6)]));
  T_dense_local[(30)] =
      (T_dense_local[(30)] + (A_shared_local1[(2)] * B_shared_local1[(7)]));
  T_dense_local[(3)] =
      (T_dense_local[(3)] + (A_shared_local1[(3)] * B_shared_local1[(0)]));
  T_dense_local[(7)] =
      (T_dense_local[(7)] + (A_shared_local1[(3)] * B_shared_local1[(1)]));
  T_dense_local[(11)] =
      (T_dense_local[(11)] + (A_shared_local1[(3)] * B_shared_local1[(2)]));
  T_dense_local[(15)] =
      (T_dense_local[(15)] + (A_shared_local1[(3)] * B_shared_local1[(3)]));
  T_dense_local[(19)] =
      (T_dense_local[(19)] + (A_shared_local1[(3)] * B_shared_local1[(4)]));
  T_dense_local[(23)] =
      (T_dense_local[(23)] + (A_shared_local1[(3)] * B_shared_local1[(5)]));
  T_dense_local[(27)] =
      (T_dense_local[(27)] + (A_shared_local1[(3)] * B_shared_local1[(6)]));
  T_dense_local[(31)] =
      (T_dense_local[(31)] + (A_shared_local1[(3)] * B_shared_local1[(7)]));
  A_shared_local1[(0)] = A_shared[(((((int)threadIdx.x) * 8) + 518))];
  A_shared_local1[(1)] = A_shared[(((((int)threadIdx.x) * 8) + 646))];
  A_shared_local1[(2)] = A_shared[(((((int)threadIdx.x) * 8) + 774))];
  A_shared_local1[(3)] = A_shared[(((((int)threadIdx.x) * 8) + 902))];
  B_shared_local1[(0)] = B_shared[(((((int)threadIdx.y) * 8) + 1030))];
  B_shared_local1[(1)] = B_shared[(((((int)threadIdx.y) * 8) + 1158))];
  B_shared_local1[(2)] = B_shared[(((((int)threadIdx.y) * 8) + 1286))];
  B_shared_local1[(3)] = B_shared[(((((int)threadIdx.y) * 8) + 1414))];
  B_shared_local1[(4)] = B_shared[(((((int)threadIdx.y) * 8) + 1542))];
  B_shared_local1[(5)] = B_shared[(((((int)threadIdx.y) * 8) + 1670))];
  B_shared_local1[(6)] = B_shared[(((((int)threadIdx.y) * 8) + 1798))];
  B_shared_local1[(7)] = B_shared[(((((int)threadIdx.y) * 8) + 1926))];
  T_dense_local[(0)] =
      (T_dense_local[(0)] + (A_shared_local1[(0)] * B_shared_local1[(0)]));
  T_dense_local[(4)] =
      (T_dense_local[(4)] + (A_shared_local1[(0)] * B_shared_local1[(1)]));
  T_dense_local[(8)] =
      (T_dense_local[(8)] + (A_shared_local1[(0)] * B_shared_local1[(2)]));
  T_dense_local[(12)] =
      (T_dense_local[(12)] + (A_shared_local1[(0)] * B_shared_local1[(3)]));
  T_dense_local[(16)] =
      (T_dense_local[(16)] + (A_shared_local1[(0)] * B_shared_local1[(4)]));
  T_dense_local[(20)] =
      (T_dense_local[(20)] + (A_shared_local1[(0)] * B_shared_local1[(5)]));
  T_dense_local[(24)] =
      (T_dense_local[(24)] + (A_shared_local1[(0)] * B_shared_local1[(6)]));
  T_dense_local[(28)] =
      (T_dense_local[(28)] + (A_shared_local1[(0)] * B_shared_local1[(7)]));
  T_dense_local[(1)] =
      (T_dense_local[(1)] + (A_shared_local1[(1)] * B_shared_local1[(0)]));
  T_dense_local[(5)] =
      (T_dense_local[(5)] + (A_shared_local1[(1)] * B_shared_local1[(1)]));
  T_dense_local[(9)] =
      (T_dense_local[(9)] + (A_shared_local1[(1)] * B_shared_local1[(2)]));
  T_dense_local[(13)] =
      (T_dense_local[(13)] + (A_shared_local1[(1)] * B_shared_local1[(3)]));
  T_dense_local[(17)] =
      (T_dense_local[(17)] + (A_shared_local1[(1)] * B_shared_local1[(4)]));
  T_dense_local[(21)] =
      (T_dense_local[(21)] + (A_shared_local1[(1)] * B_shared_local1[(5)]));
  T_dense_local[(25)] =
      (T_dense_local[(25)] + (A_shared_local1[(1)] * B_shared_local1[(6)]));
  T_dense_local[(29)] =
      (T_dense_local[(29)] + (A_shared_local1[(1)] * B_shared_local1[(7)]));
  T_dense_local[(2)] =
      (T_dense_local[(2)] + (A_shared_local1[(2)] * B_shared_local1[(0)]));
  T_dense_local[(6)] =
      (T_dense_local[(6)] + (A_shared_local1[(2)] * B_shared_local1[(1)]));
  T_dense_local[(10)] =
      (T_dense_local[(10)] + (A_shared_local1[(2)] * B_shared_local1[(2)]));
  T_dense_local[(14)] =
      (T_dense_local[(14)] + (A_shared_local1[(2)] * B_shared_local1[(3)]));
  T_dense_local[(18)] =
      (T_dense_local[(18)] + (A_shared_local1[(2)] * B_shared_local1[(4)]));
  T_dense_local[(22)] =
      (T_dense_local[(22)] + (A_shared_local1[(2)] * B_shared_local1[(5)]));
  T_dense_local[(26)] =
      (T_dense_local[(26)] + (A_shared_local1[(2)] * B_shared_local1[(6)]));
  T_dense_local[(30)] =
      (T_dense_local[(30)] + (A_shared_local1[(2)] * B_shared_local1[(7)]));
  T_dense_local[(3)] =
      (T_dense_local[(3)] + (A_shared_local1[(3)] * B_shared_local1[(0)]));
  T_dense_local[(7)] =
      (T_dense_local[(7)] + (A_shared_local1[(3)] * B_shared_local1[(1)]));
  T_dense_local[(11)] =
      (T_dense_local[(11)] + (A_shared_local1[(3)] * B_shared_local1[(2)]));
  T_dense_local[(15)] =
      (T_dense_local[(15)] + (A_shared_local1[(3)] * B_shared_local1[(3)]));
  T_dense_local[(19)] =
      (T_dense_local[(19)] + (A_shared_local1[(3)] * B_shared_local1[(4)]));
  T_dense_local[(23)] =
      (T_dense_local[(23)] + (A_shared_local1[(3)] * B_shared_local1[(5)]));
  T_dense_local[(27)] =
      (T_dense_local[(27)] + (A_shared_local1[(3)] * B_shared_local1[(6)]));
  T_dense_local[(31)] =
      (T_dense_local[(31)] + (A_shared_local1[(3)] * B_shared_local1[(7)]));
  A_shared_local1[(0)] = A_shared[(((((int)threadIdx.x) * 8) + 519))];
  A_shared_local1[(1)] = A_shared[(((((int)threadIdx.x) * 8) + 647))];
  A_shared_local1[(2)] = A_shared[(((((int)threadIdx.x) * 8) + 775))];
  A_shared_local1[(3)] = A_shared[(((((int)threadIdx.x) * 8) + 903))];
  B_shared_local1[(0)] = B_shared[(((((int)threadIdx.y) * 8) + 1031))];
  B_shared_local1[(1)] = B_shared[(((((int)threadIdx.y) * 8) + 1159))];
  B_shared_local1[(2)] = B_shared[(((((int)threadIdx.y) * 8) + 1287))];
  B_shared_local1[(3)] = B_shared[(((((int)threadIdx.y) * 8) + 1415))];
  B_shared_local1[(4)] = B_shared[(((((int)threadIdx.y) * 8) + 1543))];
  B_shared_local1[(5)] = B_shared[(((((int)threadIdx.y) * 8) + 1671))];
  B_shared_local1[(6)] = B_shared[(((((int)threadIdx.y) * 8) + 1799))];
  B_shared_local1[(7)] = B_shared[(((((int)threadIdx.y) * 8) + 1927))];
  T_dense_local[(0)] =
      (T_dense_local[(0)] + (A_shared_local1[(0)] * B_shared_local1[(0)]));
  T_dense_local[(4)] =
      (T_dense_local[(4)] + (A_shared_local1[(0)] * B_shared_local1[(1)]));
  T_dense_local[(8)] =
      (T_dense_local[(8)] + (A_shared_local1[(0)] * B_shared_local1[(2)]));
  T_dense_local[(12)] =
      (T_dense_local[(12)] + (A_shared_local1[(0)] * B_shared_local1[(3)]));
  T_dense_local[(16)] =
      (T_dense_local[(16)] + (A_shared_local1[(0)] * B_shared_local1[(4)]));
  T_dense_local[(20)] =
      (T_dense_local[(20)] + (A_shared_local1[(0)] * B_shared_local1[(5)]));
  T_dense_local[(24)] =
      (T_dense_local[(24)] + (A_shared_local1[(0)] * B_shared_local1[(6)]));
  T_dense_local[(28)] =
      (T_dense_local[(28)] + (A_shared_local1[(0)] * B_shared_local1[(7)]));
  T_dense_local[(1)] =
      (T_dense_local[(1)] + (A_shared_local1[(1)] * B_shared_local1[(0)]));
  T_dense_local[(5)] =
      (T_dense_local[(5)] + (A_shared_local1[(1)] * B_shared_local1[(1)]));
  T_dense_local[(9)] =
      (T_dense_local[(9)] + (A_shared_local1[(1)] * B_shared_local1[(2)]));
  T_dense_local[(13)] =
      (T_dense_local[(13)] + (A_shared_local1[(1)] * B_shared_local1[(3)]));
  T_dense_local[(17)] =
      (T_dense_local[(17)] + (A_shared_local1[(1)] * B_shared_local1[(4)]));
  T_dense_local[(21)] =
      (T_dense_local[(21)] + (A_shared_local1[(1)] * B_shared_local1[(5)]));
  T_dense_local[(25)] =
      (T_dense_local[(25)] + (A_shared_local1[(1)] * B_shared_local1[(6)]));
  T_dense_local[(29)] =
      (T_dense_local[(29)] + (A_shared_local1[(1)] * B_shared_local1[(7)]));
  T_dense_local[(2)] =
      (T_dense_local[(2)] + (A_shared_local1[(2)] * B_shared_local1[(0)]));
  T_dense_local[(6)] =
      (T_dense_local[(6)] + (A_shared_local1[(2)] * B_shared_local1[(1)]));
  T_dense_local[(10)] =
      (T_dense_local[(10)] + (A_shared_local1[(2)] * B_shared_local1[(2)]));
  T_dense_local[(14)] =
      (T_dense_local[(14)] + (A_shared_local1[(2)] * B_shared_local1[(3)]));
  T_dense_local[(18)] =
      (T_dense_local[(18)] + (A_shared_local1[(2)] * B_shared_local1[(4)]));
  T_dense_local[(22)] =
      (T_dense_local[(22)] + (A_shared_local1[(2)] * B_shared_local1[(5)]));
  T_dense_local[(26)] =
      (T_dense_local[(26)] + (A_shared_local1[(2)] * B_shared_local1[(6)]));
  T_dense_local[(30)] =
      (T_dense_local[(30)] + (A_shared_local1[(2)] * B_shared_local1[(7)]));
  T_dense_local[(3)] =
      (T_dense_local[(3)] + (A_shared_local1[(3)] * B_shared_local1[(0)]));
  T_dense_local[(7)] =
      (T_dense_local[(7)] + (A_shared_local1[(3)] * B_shared_local1[(1)]));
  T_dense_local[(11)] =
      (T_dense_local[(11)] + (A_shared_local1[(3)] * B_shared_local1[(2)]));
  T_dense_local[(15)] =
      (T_dense_local[(15)] + (A_shared_local1[(3)] * B_shared_local1[(3)]));
  T_dense_local[(19)] =
      (T_dense_local[(19)] + (A_shared_local1[(3)] * B_shared_local1[(4)]));
  T_dense_local[(23)] =
      (T_dense_local[(23)] + (A_shared_local1[(3)] * B_shared_local1[(5)]));
  T_dense_local[(27)] =
      (T_dense_local[(27)] + (A_shared_local1[(3)] * B_shared_local1[(6)]));
  T_dense_local[(31)] =
      (T_dense_local[(31)] + (A_shared_local1[(3)] * B_shared_local1[(7)]));
  ((float *)
       T_dense)[((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                  ((int)threadIdx.y)))] = T_dense_local[(0)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  16))] = T_dense_local[(4)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32))] = T_dense_local[(8)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  48))] = T_dense_local[(12)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  64))] = T_dense_local[(16)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  80))] = T_dense_local[(20)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  96))] = T_dense_local[(24)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  112))] = T_dense_local[(28)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32768))] = T_dense_local[(1)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32784))] = T_dense_local[(5)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32800))] = T_dense_local[(9)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32816))] = T_dense_local[(13)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32832))] = T_dense_local[(17)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32848))] = T_dense_local[(21)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32864))] = T_dense_local[(25)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  32880))] = T_dense_local[(29)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  65536))] = T_dense_local[(2)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  65552))] = T_dense_local[(6)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  65568))] = T_dense_local[(10)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  65584))] = T_dense_local[(14)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  65600))] = T_dense_local[(18)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  65616))] = T_dense_local[(22)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  65632))] = T_dense_local[(26)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  65648))] = T_dense_local[(30)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  98304))] = T_dense_local[(3)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  98320))] = T_dense_local[(7)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  98336))] = T_dense_local[(11)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  98352))] = T_dense_local[(15)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  98368))] = T_dense_local[(19)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  98384))] = T_dense_local[(23)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  98400))] = T_dense_local[(27)];
  ((float *)
       T_dense)[(((((((int)threadIdx.x) * 2048) + (((int)blockIdx.y) * 128)) +
                   ((int)threadIdx.y)) +
                  98416))] = T_dense_local[(31)];
}

__global__ void matmul_64_16384_512_kernel(void *__restrict__ A,
                                           void *__restrict__ B,
                                           void *__restrict__ compute) {
  float compute_local[64];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[2048];
  float A_shared_local[32];
  float B_shared_local[2];
  for (int i_c_init = 0; i_c_init < 32; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      compute_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    A_shared[(((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)))] =
        ((float *)A)[((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                       ((int)threadIdx.x)))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 32))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       512))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 64))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       1024))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 96))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       1536))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 128))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       2048))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 160))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       2560))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 192))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       3072))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 224))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       3584))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 256))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       4096))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 288))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       4608))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 320))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       5120))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 352))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       5632))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 384))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       6144))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 416))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       6656))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 448))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       7168))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 480))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       7680))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 512))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       8192))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 544))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       8704))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 576))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       9216))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 608))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       9728))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 640))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       10240))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 672))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       10752))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 704))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       11264))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 736))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       11776))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 768))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       12288))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 800))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       12800))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 832))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       13312))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 864))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       13824))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 896))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       14336))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 928))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       14848))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 960))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       15360))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 992))] =
        ((float *)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) +
                        ((int)threadIdx.x)) +
                       15872))];
    B_shared[(((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)))] =
        ((float *)B)[(
            ((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
              (k_outer * 32)) +
             ((int)threadIdx.x)))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 32))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             512))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 64))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             1024))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 96))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             1536))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 128))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             2048))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 160))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             2560))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 192))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             3072))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 224))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             3584))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 256))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             4096))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 288))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             4608))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 320))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             5120))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 352))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             5632))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 384))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             6144))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 416))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             6656))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 448))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             7168))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 480))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             7680))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 512))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             8192))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 544))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             8704))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 576))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             9216))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 608))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             9728))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 640))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             10240))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 672))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             10752))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 704))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             11264))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 736))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             11776))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 768))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             12288))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 800))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             12800))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 832))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             13312))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 864))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             13824))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 896))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             14336))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 928))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             14848))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 960))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             15360))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 992))] =
        ((float *)B)[(
            (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
               (k_outer * 32)) +
              ((int)threadIdx.x)) +
             15872))];
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 1024) + k_inner))];
      A_shared_local[(1)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 32))];
      A_shared_local[(2)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 64))];
      A_shared_local[(3)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 96))];
      A_shared_local[(4)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 128))];
      A_shared_local[(5)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 160))];
      A_shared_local[(6)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 192))];
      A_shared_local[(7)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 224))];
      A_shared_local[(8)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 256))];
      A_shared_local[(9)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 288))];
      A_shared_local[(10)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 320))];
      A_shared_local[(11)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 352))];
      A_shared_local[(12)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 384))];
      A_shared_local[(13)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 416))];
      A_shared_local[(14)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 448))];
      A_shared_local[(15)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 480))];
      A_shared_local[(16)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 512))];
      A_shared_local[(17)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 544))];
      A_shared_local[(18)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 576))];
      A_shared_local[(19)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 608))];
      A_shared_local[(20)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 640))];
      A_shared_local[(21)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 672))];
      A_shared_local[(22)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 704))];
      A_shared_local[(23)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 736))];
      A_shared_local[(24)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 768))];
      A_shared_local[(25)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 800))];
      A_shared_local[(26)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 832))];
      A_shared_local[(27)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 864))];
      A_shared_local[(28)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 896))];
      A_shared_local[(29)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 928))];
      A_shared_local[(30)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 960))];
      A_shared_local[(31)] =
          A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 992))];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      B_shared_local[(1)] =
          B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 32))];
      for (int i_c = 0; i_c < 32; ++i_c) {
        compute_local[((i_c * 2))] =
            (compute_local[((i_c * 2))] +
             (A_shared_local[(i_c)] * B_shared_local[(0)]));
        compute_local[(((i_c * 2) + 1))] =
            (compute_local[(((i_c * 2) + 1))] +
             (A_shared_local[(i_c)] * B_shared_local[(1)]));
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 32; ++i_inner_inner) {
    ((float *)
         compute)[(((((((int)threadIdx.y) * 524288) + (i_inner_inner * 16384)) +
                     (((int)blockIdx.x) * 64)) +
                    (((int)threadIdx.x) * 2)))] =
        compute_local[((i_inner_inner * 2))];
    ((float *)compute)[(
        (((((((int)threadIdx.y) * 524288) + (i_inner_inner * 16384)) +
           (((int)blockIdx.x) * 64)) +
          (((int)threadIdx.x) * 2)) +
         1))] = compute_local[(((i_inner_inner * 2) + 1))];
  }
}
