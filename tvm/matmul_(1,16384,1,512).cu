extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
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
      A_shared[(((((int)threadIdx.y) * 128) + (ax1_inner * 32)))] = ((float*)A)[(((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 1))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 1))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 2))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 2))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 3))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 3))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 4))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 4))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 5))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 5))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 6))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 6))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 7))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 7))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 8))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 8))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 9))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 9))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 10))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 10))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 11))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 11))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 12))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 12))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 13))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 13))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 14))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 14))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 15))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 15))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 16))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 16))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 17))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 17))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 18))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 18))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 19))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 19))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 20))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 20))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 21))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 21))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 22))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 22))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 23))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 23))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 24))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 24))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 25))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 25))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 26))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 26))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 27))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 27))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 28))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 28))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 29))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 29))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 30))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 30))];
      A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + 31))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + 31))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[((((int)threadIdx.y) * 32))] = ((float*)B)[(((((int)threadIdx.y) * 512) + (k_outer * 32)))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 1))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 1))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 2))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 2))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 3))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 3))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 4))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 4))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 5))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 5))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 6))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 6))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 7))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 7))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 8))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 8))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 9))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 9))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 10))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 10))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 11))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 11))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 12))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 12))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 13))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 13))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 14))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 14))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 15))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 15))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 16))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 16))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 17))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 17))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 18))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 18))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 19))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 19))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 20))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 20))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 21))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 21))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 22))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 22))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 23))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 23))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 24))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 24))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 25))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 25))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 26))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 26))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 27))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 27))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 28))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 28))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 29))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 29))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 30))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 30))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 32) + 31))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + 31))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 128) + k_inner))];
      A_shared_local[(1)] = A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 32))];
      A_shared_local[(2)] = A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 64))];
      A_shared_local[(3)] = A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 96))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    }
  }
  ((float*)compute)[(((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)))] = compute_local[(0)];
  ((float*)compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + 1))] = compute_local[(1)];
  ((float*)compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + 2))] = compute_local[(2)];
  ((float*)compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + 3))] = compute_local[(3)];
}

