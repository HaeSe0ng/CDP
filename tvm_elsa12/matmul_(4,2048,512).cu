extern "C" __global__ void
default_function_kernel0(void *__restrict__ A, void *__restrict__ B,
                         void *__restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[128];
  __shared__ float B_shared[4096];
  float A_shared_local[2];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
      if (((int)threadIdx.x) < 32) {
        A_shared[((((((int)threadIdx.y) * 64) + (ax1_inner * 32)) +
                   ((int)threadIdx.x)))] =
            ((float *)A)[(((((((int)threadIdx.y) * 1024) + (ax1_inner * 512)) +
                            (k_outer * 32)) +
                           ((int)threadIdx.x)))];
      }
    }
    for (int ax1_inner1 = 0; ax1_inner1 < 64; ++ax1_inner1) {
      if (((int)threadIdx.x) < 32) {
        B_shared[((((((int)threadIdx.y) * 2048) + (ax1_inner1 * 32)) +
                   ((int)threadIdx.x)))] =
            ((float *)B)[((
                ((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 32768)) +
                  (ax1_inner1 * 512)) +
                 (k_outer * 32)) +
                ((int)threadIdx.x)))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        A_shared_local[(ax1)] =
            A_shared[((((((int)threadIdx.y) * 64) + (ax1 * 32)) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 32) + k_inner))];
      for (int i_c = 0; i_c < 2; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] +
                                (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
    ((float *)
         compute)[(((((((int)threadIdx.y) * 4096) + (i_inner_inner * 2048)) +
                     (((int)blockIdx.x) * 128)) +
                    ((int)threadIdx.x)))] = compute_local[(i_inner_inner)];
  }
}
