extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[32];
  __shared__ float B_shared[1024];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = ((float*)A)[(((k_outer * 32) + ((int)threadIdx.x)))];
    B_shared[(((int)threadIdx.x))] = ((float*)B)[((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)))];
    B_shared[((((int)threadIdx.x) + 32))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 1024))];
    B_shared[((((int)threadIdx.x) + 64))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 2048))];
    B_shared[((((int)threadIdx.x) + 96))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 3072))];
    B_shared[((((int)threadIdx.x) + 128))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 4096))];
    B_shared[((((int)threadIdx.x) + 160))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 5120))];
    B_shared[((((int)threadIdx.x) + 192))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 6144))];
    B_shared[((((int)threadIdx.x) + 224))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 7168))];
    B_shared[((((int)threadIdx.x) + 256))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 8192))];
    B_shared[((((int)threadIdx.x) + 288))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 9216))];
    B_shared[((((int)threadIdx.x) + 320))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 10240))];
    B_shared[((((int)threadIdx.x) + 352))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 11264))];
    B_shared[((((int)threadIdx.x) + 384))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 12288))];
    B_shared[((((int)threadIdx.x) + 416))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 13312))];
    B_shared[((((int)threadIdx.x) + 448))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 14336))];
    B_shared[((((int)threadIdx.x) + 480))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 15360))];
    B_shared[((((int)threadIdx.x) + 512))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 16384))];
    B_shared[((((int)threadIdx.x) + 544))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 17408))];
    B_shared[((((int)threadIdx.x) + 576))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 18432))];
    B_shared[((((int)threadIdx.x) + 608))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 19456))];
    B_shared[((((int)threadIdx.x) + 640))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 20480))];
    B_shared[((((int)threadIdx.x) + 672))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 21504))];
    B_shared[((((int)threadIdx.x) + 704))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 22528))];
    B_shared[((((int)threadIdx.x) + 736))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 23552))];
    B_shared[((((int)threadIdx.x) + 768))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 24576))];
    B_shared[((((int)threadIdx.x) + 800))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 25600))];
    B_shared[((((int)threadIdx.x) + 832))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 26624))];
    B_shared[((((int)threadIdx.x) + 864))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 27648))];
    B_shared[((((int)threadIdx.x) + 896))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 28672))];
    B_shared[((((int)threadIdx.x) + 928))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 29696))];
    B_shared[((((int)threadIdx.x) + 960))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 30720))];
    B_shared[((((int)threadIdx.x) + 992))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (k_outer * 32)) + ((int)threadIdx.x)) + 31744))];
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(k_inner)];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 32) + k_inner))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float*)compute)[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))] = compute_local[(0)];
}

