extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[8];
  __shared__ float A_shared[128];
  __shared__ float B_shared[2048];
  float A_shared_local[4];
  float B_shared_local[2];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      compute_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = ((float*)A)[(((k_outer * 32) + ((int)threadIdx.x)))];
    A_shared[((((int)threadIdx.x) + 32))] = ((float*)A)[((((k_outer * 32) + ((int)threadIdx.x)) + 512))];
    A_shared[((((int)threadIdx.x) + 64))] = ((float*)A)[((((k_outer * 32) + ((int)threadIdx.x)) + 1024))];
    A_shared[((((int)threadIdx.x) + 96))] = ((float*)A)[((((k_outer * 32) + ((int)threadIdx.x)) + 1536))];
    for (int ax1_inner = 0; ax1_inner < 64; ++ax1_inner) {
      B_shared[(((ax1_inner * 32) + ((int)threadIdx.x)))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (ax1_inner * 512)) + (k_outer * 32)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      A_shared_local[(1)] = A_shared[((k_inner + 32))];
      A_shared_local[(2)] = A_shared[((k_inner + 64))];
      A_shared_local[(3)] = A_shared[((k_inner + 96))];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      B_shared_local[(1)] = B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 32))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(1)] * B_shared_local[(1)]));
      compute_local[(4)] = (compute_local[(4)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(5)] = (compute_local[(5)] + (A_shared_local[(2)] * B_shared_local[(1)]));
      compute_local[(6)] = (compute_local[(6)] + (A_shared_local[(3)] * B_shared_local[(0)]));
      compute_local[(7)] = (compute_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    }
  }
  ((float*)compute)[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  ((float*)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  ((float*)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) + 16384))] = compute_local[(2)];
  ((float*)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) + 16385))] = compute_local[(3)];
  ((float*)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) + 32768))] = compute_local[(4)];
  ((float*)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) + 32769))] = compute_local[(5)];
  ((float*)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) + 49152))] = compute_local[(6)];
  ((float*)compute)[((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) * 2)) + 49153))] = compute_local[(7)];
}

