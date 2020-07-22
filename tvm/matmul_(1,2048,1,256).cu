extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[512];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 4; ++k_outer) {
    __syncthreads();
    for (int ax2_inner = 0; ax2_inner < 64; ++ax2_inner) {
      A_shared[(((((int)threadIdx.y) * 64) + ax2_inner))] = ((float*)A)[(((((((int)blockIdx.y) * 2048) + (((int)threadIdx.y) * 256)) + (k_outer * 64)) + ax2_inner))];
    }
    for (int ax2_inner1 = 0; ax2_inner1 < 64; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 64) + ax2_inner1))] = ((float*)B)[((((((int)threadIdx.y) * 256) + (k_outer * 64)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 64; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 64) + k_inner))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float*)compute)[(((((int)blockIdx.y) * 8) + ((int)threadIdx.y)))] = compute_local[(0)];
}

