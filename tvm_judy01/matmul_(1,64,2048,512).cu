extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
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
      A_shared[((((((int)threadIdx.y) * 512) + (ax1_inner * 16)) + ((int)threadIdx.x)))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (ax1_inner * 512)) + (k_outer * 16)) + ((int)threadIdx.x)))];
    }
    #pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      B_shared[((((((int)threadIdx.y) * 128) + (ax1_inner1 * 16)) + ((int)threadIdx.x)))] = ((float*)B)[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 4096)) + (ax1_inner1 * 512)) + (k_outer * 16)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 32; ++ax1) {
        A_shared_local[(ax1)] = A_shared[((((((int)threadIdx.y) * 512) + (ax1 * 16)) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 16) + k_inner))];
      #pragma unroll
      for (int i_c = 0; i_c < 32; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] + (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
  #pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 32; ++i_inner_inner) {
    ((float*)compute)[(((((((int)threadIdx.y) * 65536) + (i_inner_inner * 2048)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)))] = compute_local[(i_inner_inner)];
  }
}

