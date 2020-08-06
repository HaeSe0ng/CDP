extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ T_dense) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    T_dense_rf[(0)] = (T_dense_rf[(0)] + (((float*)A)[((((((int)blockIdx.y) * 512) + (k_outer * 64)) + ((int)threadIdx.x)))] * ((float*)B)[((((((int)blockIdx.x) * 512) + (k_outer * 64)) + ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    ((float*)T_dense)[(((((int)blockIdx.y) * 16384) + ((int)blockIdx.x)))] = ((volatile float*)red_buf0)[(0)];
  }
}

