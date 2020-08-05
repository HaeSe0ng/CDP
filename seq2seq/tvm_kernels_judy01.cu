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
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(8, 1, 1);
    func = matmul_1_2048_512_kernel;
  } else if (M == 1 && N == 16384 && K == 512) {
    *gridDim = dim3(1024, 1, 1);
    *blockDim = dim3(16, 1, 1);
    func = matmul_1_16384_512_kernel;
  } else if (M == 4 && N == 2048 && K == 512) {
    *gridDim = dim3(128, 1, 1);
    *blockDim = dim3(16, 2, 1);
    func = matmul_4_2048_512_kernel;
  } else if (M == 4 && N == 16384 && K == 512) {
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(32, 1, 1);
    func = matmul_4_16384_512_kernel;
  } else if (M == 16 && N == 2048 && K == 512) {
    *gridDim = dim3(128, 1, 1);
    *blockDim = dim3(16, 4, 1);
    func = matmul_16_2048_512_kernel;
  } else if (M == 16 && N == 16384 && K == 512) {
    *gridDim = dim3(256, 1, 1);
    *blockDim = dim3(32, 1, 1);
    func = matmul_16_16384_512_kernel;
  } else if (M == 64 && N == 2048 && K == 512) {
    *gridDim = dim3(128, 1, 1);
    *blockDim = dim3(16, 2, 1);
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
                                         void *__restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[8];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] =
        ((float *)A)[(((k_outer * 8) + ((int)threadIdx.x)))];
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      B_shared[(((ax1_inner * 8) + ((int)threadIdx.x)))] = ((float *)B)[(
          ((((((int)blockIdx.x) * 4096) + (ax1_inner * 512)) + (k_outer * 8)) +
           ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 8) + k_inner))];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.x) * 8) + ((int)threadIdx.x)))] =
      compute_local[(0)];
}

__global__ void matmul_1_16384_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[32];
  __shared__ float B_shared[512];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
#pragma unroll
    for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
      A_shared[(((((int)threadIdx.x) * 2) + ax2_inner))] =
          ((float *)
               A)[((((k_outer * 32) + (((int)threadIdx.x) * 2)) + ax2_inner))];
    }
    for (int ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
#pragma unroll
      for (int ax2_inner1 = 0; ax2_inner1 < 2; ++ax2_inner1) {
        B_shared[(
            (((ax1_inner * 32) + (((int)threadIdx.x) * 2)) + ax2_inner1))] =
            ((float *)B)[((((((((int)blockIdx.x) * 8192) + (ax1_inner * 512)) +
                             (k_outer * 32)) +
                            (((int)threadIdx.x) * 2)) +
                           ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 32) + k_inner))];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.x) * 16) + ((int)threadIdx.x)))] =
      compute_local[(0)];
}

__global__ void matmul_4_2048_512_kernel(void *__restrict__ A,
                                         void *__restrict__ B,
                                         void *__restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[64];
  __shared__ float B_shared[256];
  float A_shared_local[2];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
      A_shared[((((((int)threadIdx.y) * 32) + (ax1_inner * 16)) +
                 ((int)threadIdx.x)))] =
          ((float *)A)[(((((((int)threadIdx.y) * 1024) + (ax1_inner * 512)) +
                          (k_outer * 16)) +
                         ((int)threadIdx.x)))];
    }
#pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      B_shared[((((((int)threadIdx.y) * 128) + (ax1_inner1 * 16)) +
                 ((int)threadIdx.x)))] =
          ((float *)B)[(
              (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 4096)) +
                 (ax1_inner1 * 512)) +
                (k_outer * 16)) +
               ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        A_shared_local[(ax1)] =
            A_shared[((((((int)threadIdx.y) * 32) + (ax1 * 16)) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 16) + k_inner))];
#pragma unroll
      for (int i_c = 0; i_c < 2; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] +
                                (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
#pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
    ((float *)
         compute)[(((((((int)threadIdx.y) * 4096) + (i_inner_inner * 2048)) +
                     (((int)blockIdx.x) * 16)) +
                    ((int)threadIdx.x)))] = compute_local[(i_inner_inner)];
  }
}

__global__ void matmul_4_16384_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ compute) {
  float compute_local[8];
  __shared__ float A_shared[128];
  __shared__ float B_shared[2048];
  float A_shared_local[4];
  float B_shared_local[2];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      compute_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] =
        ((float *)A)[(((k_outer * 32) + ((int)threadIdx.x)))];
    A_shared[((((int)threadIdx.x) + 32))] =
        ((float *)A)[((((k_outer * 32) + ((int)threadIdx.x)) + 512))];
    A_shared[((((int)threadIdx.x) + 64))] =
        ((float *)A)[((((k_outer * 32) + ((int)threadIdx.x)) + 1024))];
    A_shared[((((int)threadIdx.x) + 96))] =
        ((float *)A)[((((k_outer * 32) + ((int)threadIdx.x)) + 1536))];
    for (int ax1_inner = 0; ax1_inner < 64; ++ax1_inner) {
      B_shared[(((ax1_inner * 32) + ((int)threadIdx.x)))] = ((float *)B)[((
          (((((int)blockIdx.x) * 32768) + (ax1_inner * 512)) + (k_outer * 32)) +
          ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      A_shared_local[(1)] = A_shared[((k_inner + 32))];
      A_shared_local[(2)] = A_shared[((k_inner + 64))];
      A_shared_local[(3)] = A_shared[((k_inner + 96))];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      B_shared_local[(1)] =
          B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 32))];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] =
          (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
      compute_local[(2)] =
          (compute_local[(2)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(3)] =
          (compute_local[(3)] + (A_shared_local[(1)] * B_shared_local[(1)]));
      compute_local[(4)] =
          (compute_local[(4)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(5)] =
          (compute_local[(5)] + (A_shared_local[(2)] * B_shared_local[(1)]));
      compute_local[(6)] =
          (compute_local[(6)] + (A_shared_local[(3)] * B_shared_local[(0)]));
      compute_local[(7)] =
          (compute_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)))] =
      compute_local[(0)];
  ((float *)
       compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) + 1))] =
      compute_local[(1)];
  ((float *)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) +
                       16384))] = compute_local[(2)];
  ((float *)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) +
                       16385))] = compute_local[(3)];
  ((float *)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) +
                       32768))] = compute_local[(4)];
  ((float *)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) +
                       32769))] = compute_local[(5)];
  ((float *)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) +
                       49152))] = compute_local[(6)];
  ((float *)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) +
                       49153))] = compute_local[(7)];
}
__global__ void matmul_16_2048_512_kernel(void *__restrict__ A,
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
__global__ void matmul_16_16384_512_kernel(void *__restrict__ A,
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
__global__ void matmul_64_2048_512_kernel(void *__restrict__ A,
                                          void *__restrict__ B,
                                          void *__restrict__ compute) {
  float compute_local[32];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[256];
  float A_shared_local[32];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 32; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 32; ++ax1_inner) {
      A_shared[((((((int)threadIdx.y) * 512) + (ax1_inner * 16)) +
                 ((int)threadIdx.x)))] =
          ((float *)A)[(((((((int)threadIdx.y) * 16384) + (ax1_inner * 512)) +
                          (k_outer * 16)) +
                         ((int)threadIdx.x)))];
    }
#pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      B_shared[((((((int)threadIdx.y) * 128) + (ax1_inner1 * 16)) +
                 ((int)threadIdx.x)))] =
          ((float *)B)[(
              (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 4096)) +
                 (ax1_inner1 * 512)) +
                (k_outer * 16)) +
               ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 32; ++ax1) {
        A_shared_local[(ax1)] =
            A_shared[((((((int)threadIdx.y) * 512) + (ax1 * 16)) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 16) + k_inner))];
#pragma unroll
      for (int i_c = 0; i_c < 32; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] +
                                (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
#pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 32; ++i_inner_inner) {
    ((float *)
         compute)[(((((((int)threadIdx.y) * 65536) + (i_inner_inner * 2048)) +
                     (((int)blockIdx.x) * 16)) +
                    ((int)threadIdx.x)))] = compute_local[(i_inner_inner)];
  }
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
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 32; ++ax1_inner) {
      A_shared[((((((int)threadIdx.y) * 1024) + (ax1_inner * 32)) +
                 ((int)threadIdx.x)))] =
          ((float *)A)[(((((((int)threadIdx.y) * 16384) + (ax1_inner * 512)) +
                          (k_outer * 32)) +
                         ((int)threadIdx.x)))];
    }
#pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 32; ++ax1_inner1) {
      B_shared[((((((int)threadIdx.y) * 1024) + (ax1_inner1 * 32)) +
                 ((int)threadIdx.x)))] =
          ((float *)B)[(
              (((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) +
                 (ax1_inner1 * 512)) +
                (k_outer * 32)) +
               ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 32; ++ax1) {
        A_shared_local[(ax1)] =
            A_shared[((((((int)threadIdx.y) * 1024) + (ax1 * 32)) + k_inner))];
      }
#pragma unroll
      for (int ax11 = 0; ax11 < 2; ++ax11) {
        B_shared_local[(ax11)] =
            B_shared[((((((int)threadIdx.x) * 64) + (ax11 * 32)) + k_inner))];
      }
#pragma unroll
      for (int i_c = 0; i_c < 32; ++i_c) {
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
  for (int i_inner_inner = 0; i_inner_inner < 32; ++i_inner_inner) {
#pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
      ((float *)compute)[(
          (((((((int)threadIdx.y) * 524288) + (i_inner_inner * 16384)) +
             (((int)blockIdx.x) * 64)) +
            (((int)threadIdx.x) * 2)) +
           j_inner_inner))] =
          compute_local[(((i_inner_inner * 2) + j_inner_inner))];
    }
  }
}
