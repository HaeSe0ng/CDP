extern "C" __global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ B, void* __restrict__ C) {
  float C_local[64];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[1024];
  float A_shared_local[8];
  float B_shared_local[8];
  float A_shared_local1[8];
  float B_shared_local1[8];
  for (int ii_c_init = 0; ii_c_init < 4; ++ii_c_init) {
    for (int jj_c_init = 0; jj_c_init < 4; ++jj_c_init) {
      C_local[(((ii_c_init * 4) + jj_c_init))] = 0.000000e+00f;
      C_local[((((ii_c_init * 4) + jj_c_init) + 32))] = 0.000000e+00f;
      C_local[((((ii_c_init * 4) + jj_c_init) + 16))] = 0.000000e+00f;
      C_local[((((ii_c_init * 4) + jj_c_init) + 48))] = 0.000000e+00f;
    }
  }
  for (int ax0_inner = 0; ax0_inner < 8; ++ax0_inner) {
    for (int ax1_inner_inner_s = 0; ax1_inner_inner_s < 4; ++ax1_inner_inner_s) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner_s) < 8) {
        A_shared[(((((((int)threadIdx.y) * 64) + (ax0_inner * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner_s))] = ((float*)A)[((((((((int)blockIdx.y) * 131072) + (((int)threadIdx.y) * 16384)) + (ax0_inner * 2048)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner_s))];
      }
    }
  }
  for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      ((float4*)(B_shared + ((((((int)threadIdx.y) * 64) + (ax1_outer * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((float4*)((float*)B + (((((((int)threadIdx.y) * 2048) + (((int)blockIdx.x) * 64)) + (ax1_outer * 32)) + (((int)threadIdx.x) * 4)))))[0];
  }
  for (int k_outer_outer = 0; k_outer_outer < 255; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner1 = 0; ax0_inner1 < 8; ++ax0_inner1) {
      for (int ax1_inner_inner_s1 = 0; ax1_inner_inner_s1 < 4; ++ax1_inner_inner_s1) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner_s1) < 8) {
          if ((((k_outer_outer * 8) + (((int)threadIdx.x) * 4)) + ax1_inner_inner_s1) < 2040) {
            A_shared[((((((((k_outer_outer + 1) & 1) * 512) + (((int)threadIdx.y) * 64)) + (ax0_inner1 * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner_s1))] = ((float*)A)[((((((((((int)blockIdx.y) * 131072) + (((int)threadIdx.y) * 16384)) + (ax0_inner1 * 2048)) + (k_outer_outer * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner_s1) + 8))];
          }
        }
      }
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
        ((float4*)(B_shared + (((((((k_outer_outer + 1) & 1) * 512) + (((int)threadIdx.y) * 64)) + (ax1_outer1 * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((float4*)((float*)B + (((((((k_outer_outer * 16384) + (((int)threadIdx.y) * 2048)) + (((int)blockIdx.x) * 64)) + (ax1_outer1 * 32)) + (((int)threadIdx.x) * 4)) + 16384))))[0];
    }
    for (int ax0 = 0; ax0 < 4; ++ax0) {
      A_shared_local[(ax0)] = A_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax0 * 8)))];
      A_shared_local[((ax0 + 4))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax0 * 8)) + 256))];
    }
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      B_shared_local[(ax1)] = B_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax1))];
      B_shared_local[((ax1 + 4))] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax1) + 32))];
    }
    for (int ii_c = 0; ii_c < 4; ++ii_c) {
      for (int jj_c = 0; jj_c < 4; ++jj_c) {
        C_local[(((ii_c * 4) + jj_c))] = (C_local[(((ii_c * 4) + jj_c))] + (A_shared_local[(ii_c)] * B_shared_local[(jj_c)]));
        C_local[((((ii_c * 4) + jj_c) + 32))] = (C_local[((((ii_c * 4) + jj_c) + 32))] + (A_shared_local[((ii_c + 4))] * B_shared_local[(jj_c)]));
        C_local[((((ii_c * 4) + jj_c) + 16))] = (C_local[((((ii_c * 4) + jj_c) + 16))] + (A_shared_local[(ii_c)] * B_shared_local[((jj_c + 4))]));
        C_local[((((ii_c * 4) + jj_c) + 48))] = (C_local[((((ii_c * 4) + jj_c) + 48))] + (A_shared_local[((ii_c + 4))] * B_shared_local[((jj_c + 4))]));
      }
    }
    for (int ax01 = 0; ax01 < 4; ++ax01) {
      A_shared_local[(ax01)] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax01 * 8)) + 1))];
      A_shared_local[((ax01 + 4))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax01 * 8)) + 257))];
    }
    for (int ax11 = 0; ax11 < 4; ++ax11) {
      B_shared_local[(ax11)] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax11) + 64))];
      B_shared_local[((ax11 + 4))] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax11) + 96))];
    }
    for (int ii_c1 = 0; ii_c1 < 4; ++ii_c1) {
      for (int jj_c1 = 0; jj_c1 < 4; ++jj_c1) {
        C_local[(((ii_c1 * 4) + jj_c1))] = (C_local[(((ii_c1 * 4) + jj_c1))] + (A_shared_local[(ii_c1)] * B_shared_local[(jj_c1)]));
        C_local[((((ii_c1 * 4) + jj_c1) + 32))] = (C_local[((((ii_c1 * 4) + jj_c1) + 32))] + (A_shared_local[((ii_c1 + 4))] * B_shared_local[(jj_c1)]));
        C_local[((((ii_c1 * 4) + jj_c1) + 16))] = (C_local[((((ii_c1 * 4) + jj_c1) + 16))] + (A_shared_local[(ii_c1)] * B_shared_local[((jj_c1 + 4))]));
        C_local[((((ii_c1 * 4) + jj_c1) + 48))] = (C_local[((((ii_c1 * 4) + jj_c1) + 48))] + (A_shared_local[((ii_c1 + 4))] * B_shared_local[((jj_c1 + 4))]));
      }
    }
    for (int ax02 = 0; ax02 < 4; ++ax02) {
      A_shared_local[(ax02)] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax02 * 8)) + 2))];
      A_shared_local[((ax02 + 4))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax02 * 8)) + 258))];
    }
    for (int ax12 = 0; ax12 < 4; ++ax12) {
      B_shared_local[(ax12)] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax12) + 128))];
      B_shared_local[((ax12 + 4))] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax12) + 160))];
    }
    for (int ii_c2 = 0; ii_c2 < 4; ++ii_c2) {
      for (int jj_c2 = 0; jj_c2 < 4; ++jj_c2) {
        C_local[(((ii_c2 * 4) + jj_c2))] = (C_local[(((ii_c2 * 4) + jj_c2))] + (A_shared_local[(ii_c2)] * B_shared_local[(jj_c2)]));
        C_local[((((ii_c2 * 4) + jj_c2) + 32))] = (C_local[((((ii_c2 * 4) + jj_c2) + 32))] + (A_shared_local[((ii_c2 + 4))] * B_shared_local[(jj_c2)]));
        C_local[((((ii_c2 * 4) + jj_c2) + 16))] = (C_local[((((ii_c2 * 4) + jj_c2) + 16))] + (A_shared_local[(ii_c2)] * B_shared_local[((jj_c2 + 4))]));
        C_local[((((ii_c2 * 4) + jj_c2) + 48))] = (C_local[((((ii_c2 * 4) + jj_c2) + 48))] + (A_shared_local[((ii_c2 + 4))] * B_shared_local[((jj_c2 + 4))]));
      }
    }
    for (int ax03 = 0; ax03 < 4; ++ax03) {
      A_shared_local[(ax03)] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax03 * 8)) + 3))];
      A_shared_local[((ax03 + 4))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax03 * 8)) + 259))];
    }
    for (int ax13 = 0; ax13 < 4; ++ax13) {
      B_shared_local[(ax13)] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax13) + 192))];
      B_shared_local[((ax13 + 4))] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax13) + 224))];
    }
    for (int ii_c3 = 0; ii_c3 < 4; ++ii_c3) {
      for (int jj_c3 = 0; jj_c3 < 4; ++jj_c3) {
        C_local[(((ii_c3 * 4) + jj_c3))] = (C_local[(((ii_c3 * 4) + jj_c3))] + (A_shared_local[(ii_c3)] * B_shared_local[(jj_c3)]));
        C_local[((((ii_c3 * 4) + jj_c3) + 32))] = (C_local[((((ii_c3 * 4) + jj_c3) + 32))] + (A_shared_local[((ii_c3 + 4))] * B_shared_local[(jj_c3)]));
        C_local[((((ii_c3 * 4) + jj_c3) + 16))] = (C_local[((((ii_c3 * 4) + jj_c3) + 16))] + (A_shared_local[(ii_c3)] * B_shared_local[((jj_c3 + 4))]));
        C_local[((((ii_c3 * 4) + jj_c3) + 48))] = (C_local[((((ii_c3 * 4) + jj_c3) + 48))] + (A_shared_local[((ii_c3 + 4))] * B_shared_local[((jj_c3 + 4))]));
      }
    }
    for (int ax04 = 0; ax04 < 4; ++ax04) {
      A_shared_local[(ax04)] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax04 * 8)) + 4))];
      A_shared_local[((ax04 + 4))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax04 * 8)) + 260))];
    }
    for (int ax14 = 0; ax14 < 4; ++ax14) {
      B_shared_local[(ax14)] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax14) + 256))];
      B_shared_local[((ax14 + 4))] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax14) + 288))];
    }
    for (int ii_c4 = 0; ii_c4 < 4; ++ii_c4) {
      for (int jj_c4 = 0; jj_c4 < 4; ++jj_c4) {
        C_local[(((ii_c4 * 4) + jj_c4))] = (C_local[(((ii_c4 * 4) + jj_c4))] + (A_shared_local[(ii_c4)] * B_shared_local[(jj_c4)]));
        C_local[((((ii_c4 * 4) + jj_c4) + 32))] = (C_local[((((ii_c4 * 4) + jj_c4) + 32))] + (A_shared_local[((ii_c4 + 4))] * B_shared_local[(jj_c4)]));
        C_local[((((ii_c4 * 4) + jj_c4) + 16))] = (C_local[((((ii_c4 * 4) + jj_c4) + 16))] + (A_shared_local[(ii_c4)] * B_shared_local[((jj_c4 + 4))]));
        C_local[((((ii_c4 * 4) + jj_c4) + 48))] = (C_local[((((ii_c4 * 4) + jj_c4) + 48))] + (A_shared_local[((ii_c4 + 4))] * B_shared_local[((jj_c4 + 4))]));
      }
    }
    for (int ax05 = 0; ax05 < 4; ++ax05) {
      A_shared_local[(ax05)] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax05 * 8)) + 5))];
      A_shared_local[((ax05 + 4))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax05 * 8)) + 261))];
    }
    for (int ax15 = 0; ax15 < 4; ++ax15) {
      B_shared_local[(ax15)] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax15) + 320))];
      B_shared_local[((ax15 + 4))] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax15) + 352))];
    }
    for (int ii_c5 = 0; ii_c5 < 4; ++ii_c5) {
      for (int jj_c5 = 0; jj_c5 < 4; ++jj_c5) {
        C_local[(((ii_c5 * 4) + jj_c5))] = (C_local[(((ii_c5 * 4) + jj_c5))] + (A_shared_local[(ii_c5)] * B_shared_local[(jj_c5)]));
        C_local[((((ii_c5 * 4) + jj_c5) + 32))] = (C_local[((((ii_c5 * 4) + jj_c5) + 32))] + (A_shared_local[((ii_c5 + 4))] * B_shared_local[(jj_c5)]));
        C_local[((((ii_c5 * 4) + jj_c5) + 16))] = (C_local[((((ii_c5 * 4) + jj_c5) + 16))] + (A_shared_local[(ii_c5)] * B_shared_local[((jj_c5 + 4))]));
        C_local[((((ii_c5 * 4) + jj_c5) + 48))] = (C_local[((((ii_c5 * 4) + jj_c5) + 48))] + (A_shared_local[((ii_c5 + 4))] * B_shared_local[((jj_c5 + 4))]));
      }
    }
    for (int ax06 = 0; ax06 < 4; ++ax06) {
      A_shared_local[(ax06)] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax06 * 8)) + 6))];
      A_shared_local[((ax06 + 4))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax06 * 8)) + 262))];
    }
    for (int ax16 = 0; ax16 < 4; ++ax16) {
      B_shared_local[(ax16)] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax16) + 384))];
      B_shared_local[((ax16 + 4))] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax16) + 416))];
    }
    for (int ii_c6 = 0; ii_c6 < 4; ++ii_c6) {
      for (int jj_c6 = 0; jj_c6 < 4; ++jj_c6) {
        C_local[(((ii_c6 * 4) + jj_c6))] = (C_local[(((ii_c6 * 4) + jj_c6))] + (A_shared_local[(ii_c6)] * B_shared_local[(jj_c6)]));
        C_local[((((ii_c6 * 4) + jj_c6) + 32))] = (C_local[((((ii_c6 * 4) + jj_c6) + 32))] + (A_shared_local[((ii_c6 + 4))] * B_shared_local[(jj_c6)]));
        C_local[((((ii_c6 * 4) + jj_c6) + 16))] = (C_local[((((ii_c6 * 4) + jj_c6) + 16))] + (A_shared_local[(ii_c6)] * B_shared_local[((jj_c6 + 4))]));
        C_local[((((ii_c6 * 4) + jj_c6) + 48))] = (C_local[((((ii_c6 * 4) + jj_c6) + 48))] + (A_shared_local[((ii_c6 + 4))] * B_shared_local[((jj_c6 + 4))]));
      }
    }
    for (int ax07 = 0; ax07 < 4; ++ax07) {
      A_shared_local[(ax07)] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax07 * 8)) + 7))];
      A_shared_local[((ax07 + 4))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax07 * 8)) + 263))];
    }
    for (int ax17 = 0; ax17 < 4; ++ax17) {
      B_shared_local[(ax17)] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax17) + 448))];
      B_shared_local[((ax17 + 4))] = B_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 4)) + ax17) + 480))];
    }
    for (int ii_c7 = 0; ii_c7 < 4; ++ii_c7) {
      for (int jj_c7 = 0; jj_c7 < 4; ++jj_c7) {
        C_local[(((ii_c7 * 4) + jj_c7))] = (C_local[(((ii_c7 * 4) + jj_c7))] + (A_shared_local[(ii_c7)] * B_shared_local[(jj_c7)]));
        C_local[((((ii_c7 * 4) + jj_c7) + 32))] = (C_local[((((ii_c7 * 4) + jj_c7) + 32))] + (A_shared_local[((ii_c7 + 4))] * B_shared_local[(jj_c7)]));
        C_local[((((ii_c7 * 4) + jj_c7) + 16))] = (C_local[((((ii_c7 * 4) + jj_c7) + 16))] + (A_shared_local[(ii_c7)] * B_shared_local[((jj_c7 + 4))]));
        C_local[((((ii_c7 * 4) + jj_c7) + 48))] = (C_local[((((ii_c7 * 4) + jj_c7) + 48))] + (A_shared_local[((ii_c7 + 4))] * B_shared_local[((jj_c7 + 4))]));
      }
    }
  }
  __syncthreads();
  for (int ax08 = 0; ax08 < 4; ++ax08) {
    A_shared_local1[(ax08)] = A_shared[((((((int)threadIdx.y) * 32) + (ax08 * 8)) + 512))];
    A_shared_local1[((ax08 + 4))] = A_shared[((((((int)threadIdx.y) * 32) + (ax08 * 8)) + 768))];
  }
  for (int ax18 = 0; ax18 < 4; ++ax18) {
    B_shared_local1[(ax18)] = B_shared[((((((int)threadIdx.x) * 4) + ax18) + 512))];
    B_shared_local1[((ax18 + 4))] = B_shared[((((((int)threadIdx.x) * 4) + ax18) + 544))];
  }
  for (int ii_c8 = 0; ii_c8 < 4; ++ii_c8) {
    for (int jj_c8 = 0; jj_c8 < 4; ++jj_c8) {
      C_local[(((ii_c8 * 4) + jj_c8))] = (C_local[(((ii_c8 * 4) + jj_c8))] + (A_shared_local1[(ii_c8)] * B_shared_local1[(jj_c8)]));
      C_local[((((ii_c8 * 4) + jj_c8) + 32))] = (C_local[((((ii_c8 * 4) + jj_c8) + 32))] + (A_shared_local1[((ii_c8 + 4))] * B_shared_local1[(jj_c8)]));
      C_local[((((ii_c8 * 4) + jj_c8) + 16))] = (C_local[((((ii_c8 * 4) + jj_c8) + 16))] + (A_shared_local1[(ii_c8)] * B_shared_local1[((jj_c8 + 4))]));
      C_local[((((ii_c8 * 4) + jj_c8) + 48))] = (C_local[((((ii_c8 * 4) + jj_c8) + 48))] + (A_shared_local1[((ii_c8 + 4))] * B_shared_local1[((jj_c8 + 4))]));
    }
  }
  for (int ax09 = 0; ax09 < 4; ++ax09) {
    A_shared_local1[(ax09)] = A_shared[((((((int)threadIdx.y) * 32) + (ax09 * 8)) + 513))];
    A_shared_local1[((ax09 + 4))] = A_shared[((((((int)threadIdx.y) * 32) + (ax09 * 8)) + 769))];
  }
  for (int ax19 = 0; ax19 < 4; ++ax19) {
    B_shared_local1[(ax19)] = B_shared[((((((int)threadIdx.x) * 4) + ax19) + 576))];
    B_shared_local1[((ax19 + 4))] = B_shared[((((((int)threadIdx.x) * 4) + ax19) + 608))];
  }
  for (int ii_c9 = 0; ii_c9 < 4; ++ii_c9) {
    for (int jj_c9 = 0; jj_c9 < 4; ++jj_c9) {
      C_local[(((ii_c9 * 4) + jj_c9))] = (C_local[(((ii_c9 * 4) + jj_c9))] + (A_shared_local1[(ii_c9)] * B_shared_local1[(jj_c9)]));
      C_local[((((ii_c9 * 4) + jj_c9) + 32))] = (C_local[((((ii_c9 * 4) + jj_c9) + 32))] + (A_shared_local1[((ii_c9 + 4))] * B_shared_local1[(jj_c9)]));
      C_local[((((ii_c9 * 4) + jj_c9) + 16))] = (C_local[((((ii_c9 * 4) + jj_c9) + 16))] + (A_shared_local1[(ii_c9)] * B_shared_local1[((jj_c9 + 4))]));
      C_local[((((ii_c9 * 4) + jj_c9) + 48))] = (C_local[((((ii_c9 * 4) + jj_c9) + 48))] + (A_shared_local1[((ii_c9 + 4))] * B_shared_local1[((jj_c9 + 4))]));
    }
  }
  for (int ax010 = 0; ax010 < 4; ++ax010) {
    A_shared_local1[(ax010)] = A_shared[((((((int)threadIdx.y) * 32) + (ax010 * 8)) + 514))];
    A_shared_local1[((ax010 + 4))] = A_shared[((((((int)threadIdx.y) * 32) + (ax010 * 8)) + 770))];
  }
  for (int ax110 = 0; ax110 < 4; ++ax110) {
    B_shared_local1[(ax110)] = B_shared[((((((int)threadIdx.x) * 4) + ax110) + 640))];
    B_shared_local1[((ax110 + 4))] = B_shared[((((((int)threadIdx.x) * 4) + ax110) + 672))];
  }
  for (int ii_c10 = 0; ii_c10 < 4; ++ii_c10) {
    for (int jj_c10 = 0; jj_c10 < 4; ++jj_c10) {
      C_local[(((ii_c10 * 4) + jj_c10))] = (C_local[(((ii_c10 * 4) + jj_c10))] + (A_shared_local1[(ii_c10)] * B_shared_local1[(jj_c10)]));
      C_local[((((ii_c10 * 4) + jj_c10) + 32))] = (C_local[((((ii_c10 * 4) + jj_c10) + 32))] + (A_shared_local1[((ii_c10 + 4))] * B_shared_local1[(jj_c10)]));
      C_local[((((ii_c10 * 4) + jj_c10) + 16))] = (C_local[((((ii_c10 * 4) + jj_c10) + 16))] + (A_shared_local1[(ii_c10)] * B_shared_local1[((jj_c10 + 4))]));
      C_local[((((ii_c10 * 4) + jj_c10) + 48))] = (C_local[((((ii_c10 * 4) + jj_c10) + 48))] + (A_shared_local1[((ii_c10 + 4))] * B_shared_local1[((jj_c10 + 4))]));
    }
  }
  for (int ax011 = 0; ax011 < 4; ++ax011) {
    A_shared_local1[(ax011)] = A_shared[((((((int)threadIdx.y) * 32) + (ax011 * 8)) + 515))];
    A_shared_local1[((ax011 + 4))] = A_shared[((((((int)threadIdx.y) * 32) + (ax011 * 8)) + 771))];
  }
  for (int ax111 = 0; ax111 < 4; ++ax111) {
    B_shared_local1[(ax111)] = B_shared[((((((int)threadIdx.x) * 4) + ax111) + 704))];
    B_shared_local1[((ax111 + 4))] = B_shared[((((((int)threadIdx.x) * 4) + ax111) + 736))];
  }
  for (int ii_c11 = 0; ii_c11 < 4; ++ii_c11) {
    for (int jj_c11 = 0; jj_c11 < 4; ++jj_c11) {
      C_local[(((ii_c11 * 4) + jj_c11))] = (C_local[(((ii_c11 * 4) + jj_c11))] + (A_shared_local1[(ii_c11)] * B_shared_local1[(jj_c11)]));
      C_local[((((ii_c11 * 4) + jj_c11) + 32))] = (C_local[((((ii_c11 * 4) + jj_c11) + 32))] + (A_shared_local1[((ii_c11 + 4))] * B_shared_local1[(jj_c11)]));
      C_local[((((ii_c11 * 4) + jj_c11) + 16))] = (C_local[((((ii_c11 * 4) + jj_c11) + 16))] + (A_shared_local1[(ii_c11)] * B_shared_local1[((jj_c11 + 4))]));
      C_local[((((ii_c11 * 4) + jj_c11) + 48))] = (C_local[((((ii_c11 * 4) + jj_c11) + 48))] + (A_shared_local1[((ii_c11 + 4))] * B_shared_local1[((jj_c11 + 4))]));
    }
  }
  for (int ax012 = 0; ax012 < 4; ++ax012) {
    A_shared_local1[(ax012)] = A_shared[((((((int)threadIdx.y) * 32) + (ax012 * 8)) + 516))];
    A_shared_local1[((ax012 + 4))] = A_shared[((((((int)threadIdx.y) * 32) + (ax012 * 8)) + 772))];
  }
  for (int ax112 = 0; ax112 < 4; ++ax112) {
    B_shared_local1[(ax112)] = B_shared[((((((int)threadIdx.x) * 4) + ax112) + 768))];
    B_shared_local1[((ax112 + 4))] = B_shared[((((((int)threadIdx.x) * 4) + ax112) + 800))];
  }
  for (int ii_c12 = 0; ii_c12 < 4; ++ii_c12) {
    for (int jj_c12 = 0; jj_c12 < 4; ++jj_c12) {
      C_local[(((ii_c12 * 4) + jj_c12))] = (C_local[(((ii_c12 * 4) + jj_c12))] + (A_shared_local1[(ii_c12)] * B_shared_local1[(jj_c12)]));
      C_local[((((ii_c12 * 4) + jj_c12) + 32))] = (C_local[((((ii_c12 * 4) + jj_c12) + 32))] + (A_shared_local1[((ii_c12 + 4))] * B_shared_local1[(jj_c12)]));
      C_local[((((ii_c12 * 4) + jj_c12) + 16))] = (C_local[((((ii_c12 * 4) + jj_c12) + 16))] + (A_shared_local1[(ii_c12)] * B_shared_local1[((jj_c12 + 4))]));
      C_local[((((ii_c12 * 4) + jj_c12) + 48))] = (C_local[((((ii_c12 * 4) + jj_c12) + 48))] + (A_shared_local1[((ii_c12 + 4))] * B_shared_local1[((jj_c12 + 4))]));
    }
  }
  for (int ax013 = 0; ax013 < 4; ++ax013) {
    A_shared_local1[(ax013)] = A_shared[((((((int)threadIdx.y) * 32) + (ax013 * 8)) + 517))];
    A_shared_local1[((ax013 + 4))] = A_shared[((((((int)threadIdx.y) * 32) + (ax013 * 8)) + 773))];
  }
  for (int ax113 = 0; ax113 < 4; ++ax113) {
    B_shared_local1[(ax113)] = B_shared[((((((int)threadIdx.x) * 4) + ax113) + 832))];
    B_shared_local1[((ax113 + 4))] = B_shared[((((((int)threadIdx.x) * 4) + ax113) + 864))];
  }
  for (int ii_c13 = 0; ii_c13 < 4; ++ii_c13) {
    for (int jj_c13 = 0; jj_c13 < 4; ++jj_c13) {
      C_local[(((ii_c13 * 4) + jj_c13))] = (C_local[(((ii_c13 * 4) + jj_c13))] + (A_shared_local1[(ii_c13)] * B_shared_local1[(jj_c13)]));
      C_local[((((ii_c13 * 4) + jj_c13) + 32))] = (C_local[((((ii_c13 * 4) + jj_c13) + 32))] + (A_shared_local1[((ii_c13 + 4))] * B_shared_local1[(jj_c13)]));
      C_local[((((ii_c13 * 4) + jj_c13) + 16))] = (C_local[((((ii_c13 * 4) + jj_c13) + 16))] + (A_shared_local1[(ii_c13)] * B_shared_local1[((jj_c13 + 4))]));
      C_local[((((ii_c13 * 4) + jj_c13) + 48))] = (C_local[((((ii_c13 * 4) + jj_c13) + 48))] + (A_shared_local1[((ii_c13 + 4))] * B_shared_local1[((jj_c13 + 4))]));
    }
  }
  for (int ax014 = 0; ax014 < 4; ++ax014) {
    A_shared_local1[(ax014)] = A_shared[((((((int)threadIdx.y) * 32) + (ax014 * 8)) + 518))];
    A_shared_local1[((ax014 + 4))] = A_shared[((((((int)threadIdx.y) * 32) + (ax014 * 8)) + 774))];
  }
  for (int ax114 = 0; ax114 < 4; ++ax114) {
    B_shared_local1[(ax114)] = B_shared[((((((int)threadIdx.x) * 4) + ax114) + 896))];
    B_shared_local1[((ax114 + 4))] = B_shared[((((((int)threadIdx.x) * 4) + ax114) + 928))];
  }
  for (int ii_c14 = 0; ii_c14 < 4; ++ii_c14) {
    for (int jj_c14 = 0; jj_c14 < 4; ++jj_c14) {
      C_local[(((ii_c14 * 4) + jj_c14))] = (C_local[(((ii_c14 * 4) + jj_c14))] + (A_shared_local1[(ii_c14)] * B_shared_local1[(jj_c14)]));
      C_local[((((ii_c14 * 4) + jj_c14) + 32))] = (C_local[((((ii_c14 * 4) + jj_c14) + 32))] + (A_shared_local1[((ii_c14 + 4))] * B_shared_local1[(jj_c14)]));
      C_local[((((ii_c14 * 4) + jj_c14) + 16))] = (C_local[((((ii_c14 * 4) + jj_c14) + 16))] + (A_shared_local1[(ii_c14)] * B_shared_local1[((jj_c14 + 4))]));
      C_local[((((ii_c14 * 4) + jj_c14) + 48))] = (C_local[((((ii_c14 * 4) + jj_c14) + 48))] + (A_shared_local1[((ii_c14 + 4))] * B_shared_local1[((jj_c14 + 4))]));
    }
  }
  for (int ax015 = 0; ax015 < 4; ++ax015) {
    A_shared_local1[(ax015)] = A_shared[((((((int)threadIdx.y) * 32) + (ax015 * 8)) + 519))];
    A_shared_local1[((ax015 + 4))] = A_shared[((((((int)threadIdx.y) * 32) + (ax015 * 8)) + 775))];
  }
  for (int ax115 = 0; ax115 < 4; ++ax115) {
    B_shared_local1[(ax115)] = B_shared[((((((int)threadIdx.x) * 4) + ax115) + 960))];
    B_shared_local1[((ax115 + 4))] = B_shared[((((((int)threadIdx.x) * 4) + ax115) + 992))];
  }
  for (int ii_c15 = 0; ii_c15 < 4; ++ii_c15) {
    for (int jj_c15 = 0; jj_c15 < 4; ++jj_c15) {
      C_local[(((ii_c15 * 4) + jj_c15))] = (C_local[(((ii_c15 * 4) + jj_c15))] + (A_shared_local1[(ii_c15)] * B_shared_local1[(jj_c15)]));
      C_local[((((ii_c15 * 4) + jj_c15) + 32))] = (C_local[((((ii_c15 * 4) + jj_c15) + 32))] + (A_shared_local1[((ii_c15 + 4))] * B_shared_local1[(jj_c15)]));
      C_local[((((ii_c15 * 4) + jj_c15) + 16))] = (C_local[((((ii_c15 * 4) + jj_c15) + 16))] + (A_shared_local1[(ii_c15)] * B_shared_local1[((jj_c15 + 4))]));
      C_local[((((ii_c15 * 4) + jj_c15) + 48))] = (C_local[((((ii_c15 * 4) + jj_c15) + 48))] + (A_shared_local1[((ii_c15 + 4))] * B_shared_local1[((jj_c15 + 4))]));
    }
  }
  for (int ii_inner_inner_inner = 0; ii_inner_inner_inner < 4; ++ii_inner_inner_inner) {
    for (int jj_inner_inner_inner = 0; jj_inner_inner_inner < 4; ++jj_inner_inner_inner) {
      ((float*)C)[(((((((((int)blockIdx.y) * 131072) + (((int)threadIdx.y) * 8192)) + (ii_inner_inner_inner * 2048)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 4)) + jj_inner_inner_inner))] = C_local[(((ii_inner_inner_inner * 4) + jj_inner_inner_inner))];
      ((float*)C)[((((((((((int)blockIdx.y) * 131072) + (((int)threadIdx.y) * 8192)) + (ii_inner_inner_inner * 2048)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 4)) + jj_inner_inner_inner) + 65536))] = C_local[((((ii_inner_inner_inner * 4) + jj_inner_inner_inner) + 32))];
      ((float*)C)[((((((((((int)blockIdx.y) * 131072) + (((int)threadIdx.y) * 8192)) + (ii_inner_inner_inner * 2048)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 4)) + jj_inner_inner_inner) + 32))] = C_local[((((ii_inner_inner_inner * 4) + jj_inner_inner_inner) + 16))];
      ((float*)C)[((((((((((int)blockIdx.y) * 131072) + (((int)threadIdx.y) * 8192)) + (ii_inner_inner_inner * 2048)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 4)) + jj_inner_inner_inner) + 65568))] = C_local[((((ii_inner_inner_inner * 4) + jj_inner_inner_inner) + 48))];
    }
  }
}

