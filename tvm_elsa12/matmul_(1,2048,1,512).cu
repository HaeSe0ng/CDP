extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[256];
  __shared__ float B_shared[32];
  float A_shared_local[4];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      for (int ax2_inner = 0; ax2_inner < 32; ++ax2_inner) {
        A_shared[((((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + ax2_inner))] = ((float*)A)[((((((((int)blockIdx.y) * 4096) + (((int)threadIdx.y) * 2048)) + (ax1_inner * 512)) + (k_outer * 32)) + ax2_inner))];
      }
    }
    for (int ax2_inner1 = 0; ax2_inner1 < 32; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 32) + ax2_inner1))] = ((float*)B)[((((((int)threadIdx.y) * 512) + (k_outer * 32)) + ax2_inner1))];
      }
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
  ((float*)compute)[(((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)))] = compute_local[(0)];
  ((float*)compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 1))] = compute_local[(1)];
  ((float*)compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 2))] = compute_local[(2)];
  ((float*)compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 3))] = compute_local[(3)];
}

