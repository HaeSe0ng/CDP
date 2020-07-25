extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
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
        A_shared[(((((((int)threadIdx.y) * 512) + (ax1_inner * 32)) + (((int)threadIdx.x) * 2)) + ax2_inner))] = ((float*)A)[(((((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 8192)) + (ax1_inner * 512)) + (k_outer * 32)) + (((int)threadIdx.x) * 2)) + ax2_inner))];
      }
    }
    for (int ax1_inner1 = 0; ax1_inner1 < 16; ++ax1_inner1) {
      #pragma unroll
      for (int ax2_inner1 = 0; ax2_inner1 < 2; ++ax2_inner1) {
        B_shared[(((((((int)threadIdx.y) * 512) + (ax1_inner1 * 32)) + (((int)threadIdx.x) * 2)) + ax2_inner1))] = ((float*)B)[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.y) * 8192)) + (ax1_inner1 * 512)) + (k_outer * 32)) + (((int)threadIdx.x) * 2)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 16; ++ax1) {
        A_shared_local[(ax1)] = A_shared[((((((int)threadIdx.y) * 512) + (ax1 * 32)) + k_inner))];
      }
      #pragma unroll
      for (int ax11 = 0; ax11 < 2; ++ax11) {
        B_shared_local[(ax11)] = B_shared[((((((int)threadIdx.x) * 64) + (ax11 * 32)) + k_inner))];
      }
      for (int i_c = 0; i_c < 16; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 2; ++j_c) {
          compute_local[(((i_c * 2) + j_c))] = (compute_local[(((i_c * 2) + j_c))] + (A_shared_local[(i_c)] * B_shared_local[(j_c)]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 16; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
      ((float*)compute)[(((((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 32768)) + (i_inner_inner * 2048)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) * 2)) + j_inner_inner))] = compute_local[(((i_inner_inner * 2) + j_inner_inner))];
    }
  }
}

