extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[8];
  __shared__ float A_shared[256];
  __shared__ float B_shared[256];
  float A_shared_local[2];
  float B_shared_local[4];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
      compute_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[(((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))] = ((float*)A)[((((((int)threadIdx.y) * 1024) + (k_outer * 16)) + (((int)threadIdx.x) * 4)))];
    A_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 1))] = ((float*)A)[(((((((int)threadIdx.y) * 1024) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    A_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 2))] = ((float*)A)[(((((((int)threadIdx.y) * 1024) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    A_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 3))] = ((float*)A)[(((((((int)threadIdx.y) * 1024) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    A_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 16))] = ((float*)A)[(((((((int)threadIdx.y) * 1024) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 512))];
    A_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 17))] = ((float*)A)[(((((((int)threadIdx.y) * 1024) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 513))];
    A_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 18))] = ((float*)A)[(((((((int)threadIdx.y) * 1024) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 514))];
    A_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 19))] = ((float*)A)[(((((((int)threadIdx.y) * 1024) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 515))];
    B_shared[(((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))] = ((float*)B)[(((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) * 4)))];
    B_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 1))] = ((float*)B)[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 1))];
    B_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 2))] = ((float*)B)[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 2))];
    B_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 3))] = ((float*)B)[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 3))];
    B_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 16))] = ((float*)B)[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 512))];
    B_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 17))] = ((float*)B)[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 513))];
    B_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 18))] = ((float*)B)[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 514))];
    B_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + 19))] = ((float*)B)[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 1024)) + (k_outer * 16)) + (((int)threadIdx.x) * 4)) + 515))];
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 32) + k_inner))];
      A_shared_local[(1)] = A_shared[((((((int)threadIdx.y) * 32) + k_inner) + 16))];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      B_shared_local[(1)] = B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 16))];
      B_shared_local[(2)] = B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 32))];
      B_shared_local[(3)] = B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 48))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(0)] * B_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(0)] * B_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(5)] = (compute_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
      compute_local[(6)] = (compute_local[(6)] + (A_shared_local[(1)] * B_shared_local[(2)]));
      compute_local[(7)] = (compute_local[(7)] + (A_shared_local[(1)] * B_shared_local[(3)]));
    }
  }
  ((float*)compute)[((((((int)threadIdx.y) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)))] = compute_local[(0)];
  ((float*)compute)[(((((((int)threadIdx.y) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 1))] = compute_local[(1)];
  ((float*)compute)[(((((((int)threadIdx.y) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 2))] = compute_local[(2)];
  ((float*)compute)[(((((((int)threadIdx.y) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 3))] = compute_local[(3)];
  ((float*)compute)[(((((((int)threadIdx.y) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 2048))] = compute_local[(4)];
  ((float*)compute)[(((((((int)threadIdx.y) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 2049))] = compute_local[(5)];
  ((float*)compute)[(((((((int)threadIdx.y) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 2050))] = compute_local[(6)];
  ((float*)compute)[(((((((int)threadIdx.y) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 4)) + 2051))] = compute_local[(7)];
}

