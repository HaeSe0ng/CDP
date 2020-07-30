extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ compute) {
  float compute_local[64];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[2048];
  float A_shared_local[32];
  float B_shared_local[2];
  for (int i_c_init = 0; i_c_init < 32; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      compute_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    A_shared[(((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)))] = ((float*)A)[((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 32))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 512))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 64))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 1024))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 96))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 1536))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 128))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 2048))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 160))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 2560))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 192))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 3072))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 224))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 3584))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 256))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 4096))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 288))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 4608))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 320))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 5120))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 352))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 5632))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 384))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 6144))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 416))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 6656))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 448))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 7168))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 480))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 7680))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 512))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 8192))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 544))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 8704))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 576))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 9216))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 608))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 9728))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 640))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 10240))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 672))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 10752))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 704))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 11264))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 736))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 11776))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 768))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 12288))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 800))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 12800))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 832))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 13312))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 864))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 13824))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 896))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 14336))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 928))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 14848))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 960))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 15360))];
    A_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 992))] = ((float*)A)[(((((((int)threadIdx.y) * 16384) + (k_outer * 32)) + ((int)threadIdx.x)) + 15872))];
    B_shared[(((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)))] = ((float*)B)[(((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 32))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 512))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 64))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 1024))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 96))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 1536))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 128))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 2048))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 160))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 2560))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 192))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 3072))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 224))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 3584))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 256))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 4096))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 288))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 4608))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 320))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 5120))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 352))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 5632))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 384))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 6144))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 416))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 6656))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 448))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 7168))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 480))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 7680))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 512))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 8192))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 544))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 8704))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 576))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 9216))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 608))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 9728))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 640))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 10240))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 672))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 10752))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 704))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 11264))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 736))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 11776))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 768))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 12288))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 800))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 12800))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 832))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 13312))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 864))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 13824))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 896))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 14336))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 928))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 14848))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 960))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 15360))];
    B_shared[((((((int)threadIdx.y) * 1024) + ((int)threadIdx.x)) + 992))] = ((float*)B)[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (k_outer * 32)) + ((int)threadIdx.x)) + 15872))];
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 1024) + k_inner))];
      A_shared_local[(1)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 32))];
      A_shared_local[(2)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 64))];
      A_shared_local[(3)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 96))];
      A_shared_local[(4)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 128))];
      A_shared_local[(5)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 160))];
      A_shared_local[(6)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 192))];
      A_shared_local[(7)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 224))];
      A_shared_local[(8)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 256))];
      A_shared_local[(9)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 288))];
      A_shared_local[(10)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 320))];
      A_shared_local[(11)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 352))];
      A_shared_local[(12)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 384))];
      A_shared_local[(13)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 416))];
      A_shared_local[(14)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 448))];
      A_shared_local[(15)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 480))];
      A_shared_local[(16)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 512))];
      A_shared_local[(17)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 544))];
      A_shared_local[(18)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 576))];
      A_shared_local[(19)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 608))];
      A_shared_local[(20)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 640))];
      A_shared_local[(21)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 672))];
      A_shared_local[(22)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 704))];
      A_shared_local[(23)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 736))];
      A_shared_local[(24)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 768))];
      A_shared_local[(25)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 800))];
      A_shared_local[(26)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 832))];
      A_shared_local[(27)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 864))];
      A_shared_local[(28)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 896))];
      A_shared_local[(29)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 928))];
      A_shared_local[(30)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 960))];
      A_shared_local[(31)] = A_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 992))];
      B_shared_local[(0)] = B_shared[(((((int)threadIdx.x) * 64) + k_inner))];
      B_shared_local[(1)] = B_shared[((((((int)threadIdx.x) * 64) + k_inner) + 32))];
      for (int i_c = 0; i_c < 32; ++i_c) {
        compute_local[((i_c * 2))] = (compute_local[((i_c * 2))] + (A_shared_local[(i_c)] * B_shared_local[(0)]));
        compute_local[(((i_c * 2) + 1))] = (compute_local[(((i_c * 2) + 1))] + (A_shared_local[(i_c)] * B_shared_local[(1)]));
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 32; ++i_inner_inner) {
    ((float*)compute)[(((((((int)threadIdx.y) * 524288) + (i_inner_inner * 16384)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)))] = compute_local[((i_inner_inner * 2))];
    ((float*)compute)[((((((((int)threadIdx.y) * 524288) + (i_inner_inner * 16384)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(((i_inner_inner * 2) + 1))];
  }
}

