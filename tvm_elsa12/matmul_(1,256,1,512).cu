extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
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
        A_shared[((((((int)threadIdx.y) * 16) + (ax1_inner * 8)) + ax2_inner))] = ((float*)A)[((((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 1024)) + (ax1_inner * 512)) + (k_outer * 8)) + ax2_inner))];
      }
    }
    #pragma unroll
    for (int ax2_inner1 = 0; ax2_inner1 < 8; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 8) + ax2_inner1))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 8)) + ax2_inner1))];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        A_shared_local[(ax1)] = A_shared[((((((int)threadIdx.y) * 16) + (ax1 * 8)) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(k_inner)];
      #pragma unroll
      for (int i_c = 0; i_c < 2; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] + (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
  #pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
    ((float*)compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 2)) + i_inner_inner))] = compute_local[(i_inner_inner)];
  }
}
