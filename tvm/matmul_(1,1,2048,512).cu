extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[4];
  __shared__ float A_shared[16];
  __shared__ float B_shared[512];
  float A_shared_local[1];
  float B_shared_local[4];
  for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
    compute_local[(j_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[((((int)threadIdx.x) * 2))] = ((float*)A)[(((k_outer * 16) + (((int)threadIdx.x) * 2)))];
    A_shared[(((((int)threadIdx.x) * 2) + 1))] = ((float*)A)[((((k_outer * 16) + (((int)threadIdx.x) * 2)) + 1))];
    for (int ax1_inner = 0; ax1_inner < 32; ++ax1_inner) {
      B_shared[(((ax1_inner * 16) + (((int)threadIdx.x) * 2)))] = ((float*)B)[(((((((int)blockIdx.x) * 16384) + (ax1_inner * 512)) + (k_outer * 16)) + (((int)threadIdx.x) * 2)))];
      B_shared[((((ax1_inner * 16) + (((int)threadIdx.x) * 2)) + 1))] = ((float*)B)[((((((((int)blockIdx.x) * 16384) + (ax1_inner * 512)) + (k_outer * 16)) + (((int)threadIdx.x) * 2)) + 1))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      B_shared_local[(1)] = B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 16))];
      B_shared_local[(2)] = B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 32))];
      B_shared_local[(3)] = B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 48))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(0)] * B_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(0)] * B_shared_local[(3)]));
    }
  }
  ((float*)compute)[(((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)))] = compute_local[(0)];
  ((float*)compute)[((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)) + 1))] = compute_local[(1)];
  ((float*)compute)[((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)) + 2))] = compute_local[(2)];
  ((float*)compute)[((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)) + 3))] = compute_local[(3)];
}

