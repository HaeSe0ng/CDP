extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[32];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    A_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)))] = ((float*)A)[((((((int)threadIdx.y) * 512) + (k_outer * 8)) + ((int)threadIdx.x)))];
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
      B_shared[((((((int)threadIdx.y) * 16) + (ax1_inner * 8)) + ((int)threadIdx.x)))] = ((float*)B)[((((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 1024)) + (ax1_inner * 512)) + (k_outer * 8)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 8) + k_inner))];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 8) + k_inner))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float*)compute)[((((((int)threadIdx.y) * 2048) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = compute_local[(0)];
}

