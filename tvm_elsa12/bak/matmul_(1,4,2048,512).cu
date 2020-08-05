extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[256];
  __shared__ float B_shared[256];
  float A_shared_local[4];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      for (int ax2_inner = 0; ax2_inner < 16; ++ax2_inner) {
        A_shared[((((ax1_inner * 64) + (((int)threadIdx.x) * 16)) + ax2_inner))] = ((float*)A)[(((((ax1_inner * 512) + (k_outer * 64)) + (((int)threadIdx.x) * 16)) + ax2_inner))];
      }
    }
    for (int ax1_inner1 = 0; ax1_inner1 < 4; ++ax1_inner1) {
      for (int ax2_inner1 = 0; ax2_inner1 < 16; ++ax2_inner1) {
        B_shared[((((ax1_inner1 * 64) + (((int)threadIdx.x) * 16)) + ax2_inner1))] = ((float*)B)[((((((((int)blockIdx.x) * 2048) + (ax1_inner1 * 512)) + (k_outer * 64)) + (((int)threadIdx.x) * 16)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 64; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        A_shared_local[(ax1)] = A_shared[(((ax1 * 64) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      #pragma unroll
      for (int i_c = 0; i_c < 4; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] + (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
  #pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 4; ++i_inner_inner) {
    ((float*)compute)[((((i_inner_inner * 2048) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = compute_local[(i_inner_inner)];
  }
}

