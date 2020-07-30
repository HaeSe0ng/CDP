extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[32];
  __shared__ float B_shared[1024];
  float A_shared_local[1];
  float B_shared_local[2];
  for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
    compute_local[(j_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
      A_shared[(((((int)threadIdx.x) * 2) + ax2_inner))] = ((float*)A)[((((k_outer * 32) + (((int)threadIdx.x) * 2)) + ax2_inner))];
    }
    for (int ax1_inner = 0; ax1_inner < 32; ++ax1_inner) {
      #pragma unroll
      for (int ax2_inner1 = 0; ax2_inner1 < 2; ++ax2_inner1) {
        B_shared[((((ax1_inner * 32) + (((int)threadIdx.x) * 2)) + ax2_inner1))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (ax1_inner * 1024)) + (k_outer * 32)) + (((int)threadIdx.x) * 2)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      #pragma unroll
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        B_shared_local[(ax1)] = B_shared[((((((int)threadIdx.x) * 64) + (ax1 * 32)) + k_inner))];
      }
      #pragma unroll
      for (int j_c = 0; j_c < 2; ++j_c) {
        compute_local[(j_c)] = (compute_local[(j_c)] + (A_shared_local[(0)] * B_shared_local[(j_c)]));
      }
    }
  }
  #pragma unroll
  for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
    ((float*)compute)[((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 2)) + j_inner_inner))] = compute_local[(j_inner_inner)];
  }
}

