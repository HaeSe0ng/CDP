#include "tvm_kernels.cuh"

#include <stdio.h>

#include <cstdlib>

#include "seq2seq.h"
#include "util.h"
// batch_matmul_bsz_M_N_K

__host__ __device__ kern_func _matmul_kernel_launch_cfg(int bsz, int M, int N,
                                                        int K, dim3 *gridDim,
                                                        dim3 *blockDim) {
  kern_func func;
  if (bsz == 1 && M == 2048 && N == 1 && K == 512) {
    *gridDim = dim3(1, 256, 1);
    *blockDim = dim3(1, 2, 1);
    func = batch_matmul_1_2048_1_512_kernel;
  } else if (bsz == 1 && M == 16384 && N == 1 && K == 512) {
    *gridDim = dim3(1, 256, 1);
    *blockDim = dim3(1, 16, 1);
    func = batch_matmul_1_16384_1_512_kernel;
  } else if (bsz == 1 && M == 256 && N == 1 && K == 512) {
    *gridDim = dim3(1, 4, 1);
    *blockDim = dim3(1, 32, 1);
    func = batch_matmul_1_256_1_512_kernel;
  } else if (bsz == 64 && M == 2048 && N == 1 && K == 512) {
    *gridDim = dim3(1, 256, 1);
    *blockDim = dim3(1, 2, 1);
    func = batch_matmul_64_2048_1_512_kernel;
  } else if (bsz == 1 && M == 1 && N == 2048 && K == 512) {
    *gridDim = dim3(64, 1, 1);
    *blockDim = dim3(8, 1, 1);
    func = batch_matmul_1_1_2048_512_kernel;
  } else if (bsz == 1 && M == 1 && N == 16384 && K == 512) {
    *gridDim = dim3(512, 1, 1);
    *blockDim = dim3(32, 1, 1);
    func = batch_matmul_1_1_16384_512_kernel;
  } else if (bsz == 1 && M == 4 && N == 2048 && K == 512) {
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(8, 4, 1);
    func = batch_matmul_1_4_2048_512_kernel;
  } else if (bsz == 1 && M == 4 && N == 16384 && K == 512) {
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(32, 1, 1);
    func = batch_matmul_1_4_16384_512_kernel;
  } else if (bsz == 1 && M == 64 && N == 2048 && K == 512) {
    *gridDim = dim3(64, 2, 1);
    *blockDim = dim3(16, 2, 1);
    func = batch_matmul_1_64_2048_512_kernel;
  } else if (bsz == 1 && M == 64 && N == 16384 && K == 512) {
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(32, 2, 1);
    func = batch_matmul_1_64_16384_512_kernel;
  } else {
    printf("batch_matmul: WRONG ARGS (bsz=%d, M=%d, N=%d, K=%d)", bsz, M, N, K);
  }
  return func;
}

__global__ void batch_matmul_64_2048_1_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[512];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[((((int)threadIdx.y) * 64))] =
        ((float *)
             A)[((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                  (k_outer * 64)))];
    A_shared[(((((int)threadIdx.y) * 64) + 1))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  1))];
    A_shared[(((((int)threadIdx.y) * 64) + 2))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  2))];
    A_shared[(((((int)threadIdx.y) * 64) + 3))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  3))];
    A_shared[(((((int)threadIdx.y) * 64) + 4))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  4))];
    A_shared[(((((int)threadIdx.y) * 64) + 5))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  5))];
    A_shared[(((((int)threadIdx.y) * 64) + 6))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  6))];
    A_shared[(((((int)threadIdx.y) * 64) + 7))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  7))];
    A_shared[(((((int)threadIdx.y) * 64) + 8))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  8))];
    A_shared[(((((int)threadIdx.y) * 64) + 9))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  9))];
    A_shared[(((((int)threadIdx.y) * 64) + 10))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  10))];
    A_shared[(((((int)threadIdx.y) * 64) + 11))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  11))];
    A_shared[(((((int)threadIdx.y) * 64) + 12))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  12))];
    A_shared[(((((int)threadIdx.y) * 64) + 13))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  13))];
    A_shared[(((((int)threadIdx.y) * 64) + 14))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  14))];
    A_shared[(((((int)threadIdx.y) * 64) + 15))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  15))];
    A_shared[(((((int)threadIdx.y) * 64) + 16))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  16))];
    A_shared[(((((int)threadIdx.y) * 64) + 17))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  17))];
    A_shared[(((((int)threadIdx.y) * 64) + 18))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  18))];
    A_shared[(((((int)threadIdx.y) * 64) + 19))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  19))];
    A_shared[(((((int)threadIdx.y) * 64) + 20))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  20))];
    A_shared[(((((int)threadIdx.y) * 64) + 21))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  21))];
    A_shared[(((((int)threadIdx.y) * 64) + 22))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  22))];
    A_shared[(((((int)threadIdx.y) * 64) + 23))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  23))];
    A_shared[(((((int)threadIdx.y) * 64) + 24))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  24))];
    A_shared[(((((int)threadIdx.y) * 64) + 25))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  25))];
    A_shared[(((((int)threadIdx.y) * 64) + 26))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  26))];
    A_shared[(((((int)threadIdx.y) * 64) + 27))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  27))];
    A_shared[(((((int)threadIdx.y) * 64) + 28))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  28))];
    A_shared[(((((int)threadIdx.y) * 64) + 29))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  29))];
    A_shared[(((((int)threadIdx.y) * 64) + 30))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  30))];
    A_shared[(((((int)threadIdx.y) * 64) + 31))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  31))];
    A_shared[(((((int)threadIdx.y) * 64) + 32))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  32))];
    A_shared[(((((int)threadIdx.y) * 64) + 33))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  33))];
    A_shared[(((((int)threadIdx.y) * 64) + 34))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  34))];
    A_shared[(((((int)threadIdx.y) * 64) + 35))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  35))];
    A_shared[(((((int)threadIdx.y) * 64) + 36))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  36))];
    A_shared[(((((int)threadIdx.y) * 64) + 37))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  37))];
    A_shared[(((((int)threadIdx.y) * 64) + 38))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  38))];
    A_shared[(((((int)threadIdx.y) * 64) + 39))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  39))];
    A_shared[(((((int)threadIdx.y) * 64) + 40))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  40))];
    A_shared[(((((int)threadIdx.y) * 64) + 41))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  41))];
    A_shared[(((((int)threadIdx.y) * 64) + 42))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  42))];
    A_shared[(((((int)threadIdx.y) * 64) + 43))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  43))];
    A_shared[(((((int)threadIdx.y) * 64) + 44))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  44))];
    A_shared[(((((int)threadIdx.y) * 64) + 45))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  45))];
    A_shared[(((((int)threadIdx.y) * 64) + 46))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  46))];
    A_shared[(((((int)threadIdx.y) * 64) + 47))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  47))];
    A_shared[(((((int)threadIdx.y) * 64) + 48))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  48))];
    A_shared[(((((int)threadIdx.y) * 64) + 49))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  49))];
    A_shared[(((((int)threadIdx.y) * 64) + 50))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  50))];
    A_shared[(((((int)threadIdx.y) * 64) + 51))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  51))];
    A_shared[(((((int)threadIdx.y) * 64) + 52))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  52))];
    A_shared[(((((int)threadIdx.y) * 64) + 53))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  53))];
    A_shared[(((((int)threadIdx.y) * 64) + 54))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  54))];
    A_shared[(((((int)threadIdx.y) * 64) + 55))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  55))];
    A_shared[(((((int)threadIdx.y) * 64) + 56))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  56))];
    A_shared[(((((int)threadIdx.y) * 64) + 57))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  57))];
    A_shared[(((((int)threadIdx.y) * 64) + 58))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  58))];
    A_shared[(((((int)threadIdx.y) * 64) + 59))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  59))];
    A_shared[(((((int)threadIdx.y) * 64) + 60))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  60))];
    A_shared[(((((int)threadIdx.y) * 64) + 61))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  61))];
    A_shared[(((((int)threadIdx.y) * 64) + 62))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  62))];
    A_shared[(((((int)threadIdx.y) * 64) + 63))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  63))];
    if (((int)threadIdx.y) < 1) {
      B_shared[((((int)threadIdx.y) * 64))] =
          ((float *)B)[(((((int)threadIdx.y) * 2048) + (k_outer * 64)))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 1))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 1))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 2))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 2))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 3))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 3))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 4))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 4))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 5))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 5))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 6))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 6))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 7))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 7))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 8))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 8))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 9))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 9))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 10))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 10))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 11))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 11))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 12))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 12))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 13))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 13))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 14))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 14))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 15))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 15))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 16))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 16))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 17))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 17))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 18))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 18))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 19))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 19))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 20))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 20))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 21))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 21))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 22))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 22))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 23))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 23))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 24))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 24))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 25))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 25))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 26))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 26))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 27))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 27))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 28))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 28))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 29))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 29))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 30))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 30))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 31))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 31))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 32))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 32))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 33))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 33))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 34))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 34))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 35))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 35))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 36))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 36))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 37))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 37))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 38))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 38))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 39))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 39))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 40))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 40))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 41))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 41))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 42))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 42))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 43))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 43))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 44))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 44))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 45))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 45))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 46))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 46))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 47))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 47))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 48))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 48))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 49))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 49))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 50))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 50))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 51))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 51))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 52))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 52))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 53))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 53))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 54))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 54))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 55))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 55))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 56))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 56))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 57))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 57))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 58))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 58))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 59))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 59))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 60))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 60))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 61))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 61))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 62))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 62))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 63))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 63))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 64; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 64) + k_inner))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.y) * 8) + ((int)threadIdx.y)))] =
      compute_local[(0)];
}

__global__ void batch_matmul_1_256_1_512_kernel(void *__restrict__ A,
                                                void *__restrict__ B,
                                                void *__restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[512];
  __shared__ float B_shared[8];
  float A_shared_local[2];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
#pragma unroll
      for (int ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
        A_shared[(
            (((((int)threadIdx.y) * 16) + (ax1_inner * 8)) + ax2_inner))] =
            ((float *)A)[(
                (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 1024)) +
                   (ax1_inner * 512)) +
                  (k_outer * 8)) +
                 ax2_inner))];
      }
    }
#pragma unroll
    for (int ax2_inner1 = 0; ax2_inner1 < 8; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 8) + ax2_inner1))] = ((float *)B)[(
            (((((int)threadIdx.y) * 512) + (k_outer * 8)) + ax2_inner1))];
      }
    }
    __syncthreads();
#pragma unroll
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        A_shared_local[(ax1)] =
            A_shared[((((((int)threadIdx.y) * 16) + (ax1 * 8)) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(k_inner)];
#pragma unroll
      for (int i_c = 0; i_c < 2; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] +
                                (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
#pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
    ((float *)compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 2)) +
                         i_inner_inner))] = compute_local[(i_inner_inner)];
  }
}

__global__ void batch_matmul_1_2048_1_256_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[512];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 4; ++k_outer) {
    __syncthreads();
    for (int ax2_inner = 0; ax2_inner < 64; ++ax2_inner) {
      A_shared[(((((int)threadIdx.y) * 64) + ax2_inner))] =
          ((float *)
               A)[(((((((int)blockIdx.y) * 2048) + (((int)threadIdx.y) * 256)) +
                     (k_outer * 64)) +
                    ax2_inner))];
    }
    for (int ax2_inner1 = 0; ax2_inner1 < 64; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 64) + ax2_inner1))] = ((float *)B)[(
            (((((int)threadIdx.y) * 256) + (k_outer * 64)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 64; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 64) + k_inner))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.y) * 8) + ((int)threadIdx.y)))] =
      compute_local[(0)];
}

__global__ void batch_matmul_1_2048_1_512_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[256];
  __shared__ float B_shared[32];
  float A_shared_local[4];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      for (int ax2_inner = 0; ax2_inner < 32; ++ax2_inner) {
        A_shared[(
            (((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + ax2_inner))] =
            ((float *)A)[(
                (((((((int)blockIdx.y) * 4096) + (((int)threadIdx.y) * 2048)) +
                   (ax1_inner * 512)) +
                  (k_outer * 32)) +
                 ax2_inner))];
      }
    }
    for (int ax2_inner1 = 0; ax2_inner1 < 32; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 32) + ax2_inner1))] = ((float *)B)[(
            (((((int)threadIdx.y) * 512) + (k_outer * 32)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 128) + k_inner))];
      A_shared_local[(1)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 32))];
      A_shared_local[(2)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 64))];
      A_shared_local[(3)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 96))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] =
          (compute_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(2)] =
          (compute_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(3)] =
          (compute_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)))] =
      compute_local[(0)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 1))] =
      compute_local[(1)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 2))] =
      compute_local[(2)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 3))] =
      compute_local[(3)];
}

__global__ void batch_matmul_1_16384_1_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[32];
  float A_shared_local[4];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      A_shared[(((((int)threadIdx.y) * 128) + (ax1_inner * 32)))] =
          ((float *)A)[(
              ((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                (ax1_inner * 512)) +
               (k_outer * 32)))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 1))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               1))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 2))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               2))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 3))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               3))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 4))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               4))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 5))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               5))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 6))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               6))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 7))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               7))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 8))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               8))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 9))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               9))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 10))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               10))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 11))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               11))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 12))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               12))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 13))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               13))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 14))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               14))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 15))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               15))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 16))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               16))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 17))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               17))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 18))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               18))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 19))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               19))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 20))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               20))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 21))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               21))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 22))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               22))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 23))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               23))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 24))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               24))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 25))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               25))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 26))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               26))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 27))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               27))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 28))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               28))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 29))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               29))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 30))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               30))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 31))] =
          ((float *)A)[(
              (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner * 512)) +
                (k_outer * 32)) +
               31))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[((((int)threadIdx.y) * 32))] =
          ((float *)B)[(((((int)threadIdx.y) * 512) + (k_outer * 32)))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 1))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 1))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 2))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 2))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 3))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 3))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 4))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 4))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 5))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 5))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 6))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 6))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 7))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 7))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 8))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 8))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 9))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 9))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 10))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 10))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 11))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 11))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 12))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 12))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 13))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 13))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 14))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 14))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 15))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 15))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 16))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 16))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 17))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 17))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 18))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 18))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 19))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 19))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 20))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 20))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 21))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 21))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 22))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 22))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 23))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 23))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 24))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 24))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 25))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 25))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 26))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 26))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 27))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 27))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 28))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 28))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 29))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 29))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 30))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 30))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 31))] =
          ((float *)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 31))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 128) + k_inner))];
      A_shared_local[(1)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 32))];
      A_shared_local[(2)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 64))];
      A_shared_local[(3)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 96))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] =
          (compute_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(2)] =
          (compute_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(3)] =
          (compute_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)))] =
      compute_local[(0)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + 1))] =
      compute_local[(1)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + 2))] =
      compute_local[(2)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + 3))] =
      compute_local[(3)];
}

__global__ void batch_matmul_1_1_2048_512_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[16];
  __shared__ float B_shared[512];
  float A_shared_local[1];
  float B_shared_local[4];
  for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
    compute_local[(j_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[((((int)threadIdx.x) * 2))] =
        ((float *)A)[(((k_outer * 16) + (((int)threadIdx.x) * 2)))];
    A_shared[(((((int)threadIdx.x) * 2) + 1))] =
        ((float *)A)[((((k_outer * 16) + (((int)threadIdx.x) * 2)) + 1))];
    for (int ax1_inner = 0; ax1_inner < 32; ++ax1_inner) {
      B_shared[(((ax1_inner * 16) + (((int)threadIdx.x) * 2)))] =
          ((float *)B)[(((((((int)blockIdx.x) * 16384) + (ax1_inner * 512)) +
                          (k_outer * 16)) +
                         (((int)threadIdx.x) * 2)))];
      B_shared[((((ax1_inner * 16) + (((int)threadIdx.x) * 2)) + 1))] =
          ((float *)B)[((((((((int)blockIdx.x) * 16384) + (ax1_inner * 512)) +
                           (k_outer * 16)) +
                          (((int)threadIdx.x) * 2)) +
                         1))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      B_shared_local[(1)] =
          B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 16))];
      B_shared_local[(2)] =
          B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 32))];
      B_shared_local[(3)] =
          B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 48))];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] =
          (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
      compute_local[(2)] =
          (compute_local[(2)] + (A_shared_local[(0)] * B_shared_local[(2)]));
      compute_local[(3)] =
          (compute_local[(3)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)))] =
      compute_local[(0)];
  ((float *)
       compute)[((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)) + 1))] =
      compute_local[(1)];
  ((float *)
       compute)[((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)) + 2))] =
      compute_local[(2)];
  ((float *)
       compute)[((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)) + 3))] =
      compute_local[(3)];
}

__global__ void batch_matmul_1_1_16384_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[32];
  __shared__ float B_shared[1024];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] =
        ((float *)A)[(((k_outer * 32) + ((int)threadIdx.x)))];
    for (int ax1_inner = 0; ax1_inner < 32; ++ax1_inner) {
      B_shared[(((ax1_inner * 32) + ((int)threadIdx.x)))] = ((float *)B)[((
          (((((int)blockIdx.x) * 16384) + (ax1_inner * 512)) + (k_outer * 32)) +
          ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 32) + k_inner))];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))] =
      compute_local[(0)];
}

__global__ void batch_matmul_1_4_2048_512_kernel(void *__restrict__ A,
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

__global__ void batch_matmul_1_4_16384_512_kernel(void *__restrict__ A,
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
__global__ void batch_matmul_1_16_2048_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[256];
  __shared__ float B_shared[256];
  float A_shared_local[4];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      A_shared[((((((int)threadIdx.y) * 64) + (ax1_inner * 16)) +
                 ((int)threadIdx.x)))] =
          ((float *)A)[(((((((int)threadIdx.y) * 2048) + (ax1_inner * 512)) +
                          (k_outer * 16)) +
                         ((int)threadIdx.x)))];
    }
#pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 4; ++ax1_inner1) {
      B_shared[((((((int)threadIdx.y) * 64) + (ax1_inner1 * 16)) +
                 ((int)threadIdx.x)))] =
          ((float *)B)[(
              (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) +
                 (ax1_inner1 * 512)) +
                (k_outer * 16)) +
               ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        A_shared_local[(ax1)] =
            A_shared[((((((int)threadIdx.y) * 64) + (ax1 * 16)) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 16) + k_inner))];
#pragma unroll
      for (int i_c = 0; i_c < 4; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] +
                                (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
#pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 4; ++i_inner_inner) {
    ((float *)
         compute)[(((((((int)threadIdx.y) * 8192) + (i_inner_inner * 2048)) +
                     (((int)blockIdx.x) * 16)) +
                    ((int)threadIdx.x)))] = compute_local[(i_inner_inner)];
  }
}
__global__ void batch_matmul_1_16_16384_512_kernel(void *__restrict__ A,
                                                   void *__restrict__ B,
                                                   void *__restrict__ compute) {
  float compute_local[32];
  __shared__ float A_shared[512];
  __shared__ float B_shared[2048];
  float A_shared_local[16];
  float B_shared_local[2];
  for (int i_c_init = 0; i_c_init < 16; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      compute_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      A_shared[(((ax1_inner * 32) + ((int)threadIdx.x)))] = ((float *)A)[(
          (((ax1_inner * 512) + (k_outer * 32)) + ((int)threadIdx.x)))];
    }
#pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 64; ++ax1_inner1) {
      B_shared[(((ax1_inner1 * 32) + ((int)threadIdx.x)))] =
          ((float *)B)[(((((((int)blockIdx.x) * 32768) + (ax1_inner1 * 512)) +
                          (k_outer * 32)) +
                         ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 16; ++ax1) {
        A_shared_local[(ax1)] = A_shared[(((ax1 * 32) + k_inner))];
      }
#pragma unroll
      for (int ax11 = 0; ax11 < 2; ++ax11) {
        B_shared_local[(ax11)] =
            B_shared[((((((int)threadIdx.x) * 64) + (ax11 * 32)) + k_inner))];
      }
#pragma unroll
      for (int i_c = 0; i_c < 16; ++i_c) {
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
  for (int i_inner_inner = 0; i_inner_inner < 16; ++i_inner_inner) {
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
__global__ void batch_matmul_1_64_2048_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute) {
  float compute_local[32];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[1024];
  float A_shared_local[16];
  float B_shared_local[2];
  for (int i_c_init = 0; i_c_init < 16; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      compute_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    for (int ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
#pragma unroll
      for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
        A_shared[(((((((int)threadIdx.y) * 512) + (ax1_inner * 32)) +
                    (((int)threadIdx.x) * 2)) +
                   ax2_inner))] =
            ((float *)A)[((
                (((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 8192)) +
                   (ax1_inner * 512)) +
                  (k_outer * 32)) +
                 (((int)threadIdx.x) * 2)) +
                ax2_inner))];
      }
    }
    for (int ax1_inner1 = 0; ax1_inner1 < 16; ++ax1_inner1) {
#pragma unroll
      for (int ax2_inner1 = 0; ax2_inner1 < 2; ++ax2_inner1) {
        B_shared[(((((((int)threadIdx.y) * 512) + (ax1_inner1 * 32)) +
                    (((int)threadIdx.x) * 2)) +
                   ax2_inner1))] =
            ((float *)B)[((
                (((((((int)blockIdx.x) * 16384) + (((int)threadIdx.y) * 8192)) +
                   (ax1_inner1 * 512)) +
                  (k_outer * 32)) +
                 (((int)threadIdx.x) * 2)) +
                ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 16; ++ax1) {
        A_shared_local[(ax1)] =
            A_shared[((((((int)threadIdx.y) * 512) + (ax1 * 32)) + k_inner))];
      }
#pragma unroll
      for (int ax11 = 0; ax11 < 2; ++ax11) {
        B_shared_local[(ax11)] =
            B_shared[((((((int)threadIdx.x) * 64) + (ax11 * 32)) + k_inner))];
      }
      for (int i_c = 0; i_c < 16; ++i_c) {
#pragma unroll
        for (int j_c = 0; j_c < 2; ++j_c) {
          compute_local[(((i_c * 2) + j_c))] =
              (compute_local[(((i_c * 2) + j_c))] +
               (A_shared_local[(i_c)] * B_shared_local[(j_c)]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 16; ++i_inner_inner) {
#pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
      ((float *)compute)[(
          ((((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 32768)) +
              (i_inner_inner * 2048)) +
             (((int)blockIdx.x) * 32)) +
            (((int)threadIdx.x) * 2)) +
           j_inner_inner))] =
          compute_local[(((i_inner_inner * 2) + j_inner_inner))];
    }
  }
}

__global__ void batch_matmul_1_64_16384_512_kernel(void *__restrict__ A,
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
