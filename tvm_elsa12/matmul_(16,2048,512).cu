
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif

extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ T_dense) {
  float T_dense_rf[1];
  float red_buf0[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    T_dense_rf[(0)] = (T_dense_rf[(0)] + (((float*)A)[((((((int)blockIdx.y) * 512) + (k_outer * 32)) + ((int)threadIdx.x)))] * ((float*)B)[((((((int)blockIdx.x) * 512) + (k_outer * 32)) + ((int)threadIdx.x)))]));
  }
  unsigned int mask[1];
  float t0[1];
  red_buf0[(0)] = T_dense_rf[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  if (((int)threadIdx.x) == 0) {
    ((float*)T_dense)[(((((int)blockIdx.y) * 2048) + ((int)blockIdx.x)))] = red_buf0[(0)];
  }
}

