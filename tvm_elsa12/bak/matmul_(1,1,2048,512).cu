extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[64];
  __shared__ float B_shared[2048];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
      A_shared[(((((int)threadIdx.x) * 2) + ax2_inner))] = ((float*)A)[((((k_outer * 64) + (((int)threadIdx.x) * 2)) + ax2_inner))];
    }
    for (int ax1_inner = 0; ax1_inner < 32; ++ax1_inner) {
      #pragma unroll
      for (int ax2_inner1 = 0; ax2_inner1 < 2; ++ax2_inner1) {
        B_shared[((((ax1_inner * 64) + (((int)threadIdx.x) * 2)) + ax2_inner1))] = ((float*)B)[((((((((int)blockIdx.x) * 16384) + (ax1_inner * 512)) + (k_outer * 64)) + (((int)threadIdx.x) * 2)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 64; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float*)compute)[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))] = compute_local[(0)];
}

