#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdlib>

#include "seq2seq.h"
#include "util.h"

#define AT_APPLY_THREADS_PER_BLOCK 512

__global__ void batch_matmul_1_256_1_512_kernel(void *__restrict__ A,
                                                void *__restrict__ B,
                                                void *__restrict__ compute);
__global__ void batch_matmul_1_2048_1_256_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute);
__global__ void batch_matmul_1_2048_1_512_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute);

template <typename T> __device__ __forceinline__ T sigmoid(T in) {
  T one = static_cast<T>(1.0);
  return one / (one + exp(-in));
}

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {}
// bias1: input_bias, bias2: hidden_bias, cx: last cell state, hsz: hidden_size
__global__ void lstm_cell_kernel(float *input, float *hidden, float *bias1,
                                 float *bias2, float *_cx, float *_hy,
                                 float *_cy, int hsz, int totalElements) {
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements; linearIndex += gridDim.x * blockDim.x) {
    int offset = (linearIndex / hsz) * 4 * hsz + linearIndex % hsz;

    float iig = input[offset + 0 * hsz];
    float ifg = input[offset + 1 * hsz];
    float icg = input[offset + 2 * hsz];
    float iog = input[offset + 3 * hsz];

    float hig = hidden[offset + 0 * hsz];
    float hfg = hidden[offset + 1 * hsz];
    float hcg = hidden[offset + 2 * hsz];
    float hog = hidden[offset + 3 * hsz];

    float cx = _cx[linearIndex];

    float *hy = &_hy[linearIndex];
    float *cy = &_cy[linearIndex];

    float b1i, b1f, b1c, b1o;
    float b2i, b2f, b2c, b2o;

    b1i = bias1[linearIndex % hsz + 0 * hsz];
    b1f = bias1[linearIndex % hsz + 1 * hsz];
    b1c = bias1[linearIndex % hsz + 2 * hsz];
    b1o = bias1[linearIndex % hsz + 3 * hsz];

    b2i = bias2[linearIndex % hsz + 0 * hsz];
    b2f = bias2[linearIndex % hsz + 1 * hsz];
    b2c = bias2[linearIndex % hsz + 2 * hsz];
    b2o = bias2[linearIndex % hsz + 3 * hsz];

    float ig, fg, cg, og;
    float f_hy, f_cy;

    ig = sigmoid(iig + hig + b1i + b2i);
    fg = sigmoid(ifg + hfg + b1f + b2f);
    cg = tanh(icg + hcg + b1c + b2c);
    og = sigmoid(iog + hog + b1o + b2o);

    f_cy = (fg * cx) + (ig * cg);
    f_hy = og * tanh(f_cy);

    *hy = f_hy;
    *cy = f_cy;
  }
}
void seq2seq_encode(float *input_d, float *hidden_d, float *w_ih_d,
                    float *w_hh_d, float *igate_d, float *hgate_d,
                    float *b_ih_d, float *b_hh_d, float *cell_d,
                    float *output_d, float *w_ho_d, int input_dim,
                    int hidden_size, int totalElements, int totalInputs,
                    int seq_length) {
  for (int i = 0; i < seq_length; i++) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 8, 1);
    batch_matmul_1_2048_1_256_kernel<<<gridDim, blockDim>>>(
        w_ih_d, input_d + input_dim * i,
        igate_d); //(4 * totalElements), 1, totalInputs
    dim3 gridDim2(1, 256, 1);
    dim3 blockDim2(1, 2, 1);
    batch_matmul_1_2048_1_512_kernel<<<gridDim2, blockDim2>>>(
        w_hh_d, hidden_d, hgate_d); // (4 * totalElements), 1, totalElements
    lstm_cell_kernel<<<totalElements / AT_APPLY_THREADS_PER_BLOCK,
                       AT_APPLY_THREADS_PER_BLOCK>>>(
        igate_d, hgate_d, b_ih_d, b_hh_d, cell_d, hidden_d, cell_d, hidden_size,
        totalElements);
  }
}
__global__ void seq2seq_decode(float *input_d, float *hidden_d, float *w_ih_d,
                               float *w_hh_d, float *igate_d, float *hgate_d,
                               float *b_ih_d, float *b_hh_d, float *cell_d,
                               float *output_d, float *w_ho_d, int input_dim,
                               int hidden_size, int totalElements,
                               int totalInputs) {
  int i = 0;
  do {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 8, 1);
    batch_matmul_1_2048_1_256_kernel<<<gridDim, blockDim>>>(
        w_ih_d, input_d + input_dim * i,
        igate_d); // (4 * totalElements), 1, totalInputs
    dim3 gridDim2(1, 256, 1);
    dim3 blockDim2(1, 2, 1);
    batch_matmul_1_2048_1_512_kernel<<<gridDim2, blockDim2>>>(
        w_hh_d, hidden_d, hgate_d); // (4 * totalElements), 1, totalElements
    lstm_cell_kernel<<<totalElements / AT_APPLY_THREADS_PER_BLOCK,
                       AT_APPLY_THREADS_PER_BLOCK>>>(
        igate_d, hgate_d, b_ih_d, b_hh_d, cell_d, hidden_d, cell_d, hidden_size,
        totalElements);
    dim3 gridDim3(1, 4, 1);
    dim3 blockDim3(1, 32, 1);
    batch_matmul_1_256_1_512_kernel<<<gridDim3, blockDim3>>>(
        w_ho_d, hidden_d,
        output_d + input_dim * i); // totalInputs, 1, totalElements
    i++;
  } while (i < 15);
}
int seq2seq_inf(float *input, float *output, int input_dim, int seq_length,
                int hidden_size, int batch_size) {
  srand(10);
  int totalElements = batch_size * hidden_size;
  int totalInputs = batch_size * input_dim;
  float *w_ih, *w_hh, *b_ih, *b_hh, *w_ho;
  float *input_d, *hidden_d, *igate_d, *hgate_d, *cell_d, *w_ih_d, *w_hh_d,
      *b_ih_d, *b_hh_d, *w_ho_d, *output_d;

  cudaMalloc((void **)&input_d, sizeof(float) * totalInputs);
  cudaMalloc((void **)&output_d, sizeof(float) * totalInputs);

  cudaMalloc((void **)&hidden_d, sizeof(float) * totalElements);
  cudaMalloc((void **)&cell_d, sizeof(float) * totalElements);

  cudaMalloc((void **)&igate_d, sizeof(float) * (4 * totalElements) * 1);
  cudaMalloc((void **)&hgate_d, sizeof(float) * (4 * totalElements) * 1);

  cudaMalloc((void **)&w_ih_d,
             sizeof(float) * (4 * totalElements) * totalInputs);
  cudaMalloc((void **)&w_hh_d,
             sizeof(float) * (4 * totalElements) * totalElements);
  cudaMalloc((void **)&b_ih_d, sizeof(float) * 4 * hidden_size);
  cudaMalloc((void **)&b_hh_d, sizeof(float) * 4 * hidden_size);
  cudaMalloc((void **)&w_ho_d, sizeof(float) * totalInputs * totalElements);

  cudaMemset(hidden_d, 0, sizeof(float) * totalElements);
  cudaMemset(cell_d, 0, sizeof(float) * totalElements);
  alloc_rand_mat(&w_ih, (4 * totalElements), totalInputs);
  alloc_rand_mat(&w_hh, (4 * totalElements), totalElements);
  alloc_rand_mat(&b_ih, 1, 4 * hidden_size);
  alloc_rand_mat(&b_hh, 1, 4 * hidden_size);
  alloc_rand_mat(&w_ho, totalInputs, totalElements);

  cudaMemcpy(input_d, input, (sizeof(float) * totalInputs),
             cudaMemcpyHostToDevice);
  cudaMemcpy(w_ih_d, w_ih, (sizeof(float) * (4 * totalElements) * totalInputs),
             cudaMemcpyHostToDevice);
  cudaMemcpy(w_hh_d, w_hh,
             (sizeof(float) * (4 * totalElements) * totalElements),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_ih_d, b_ih, (sizeof(float) * 4 * hidden_size),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_hh_d, b_hh, (sizeof(float) * 4 * hidden_size),
             cudaMemcpyHostToDevice);
  cudaMemcpy(w_ho_d, w_ho, (sizeof(float) * totalInputs * totalElements),
             cudaMemcpyHostToDevice);

  seq2seq_encode(input_d, hidden_d, w_ih_d, w_hh_d, igate_d, hgate_d, b_ih_d,
                 b_hh_d, cell_d, output_d, w_ho_d, input_dim, hidden_size,
                 totalElements, totalInputs, seq_length);
  seq2seq_decode<<<1, 1>>>(input_d, hidden_d, w_ih_d, w_hh_d, igate_d, hgate_d,
                           b_ih_d, b_hh_d, cell_d, output_d, w_ho_d, input_dim,
                           hidden_size, totalElements, totalInputs);

  cudaMemcpy(output, output_d, (sizeof(float) * totalInputs),
             cudaMemcpyDeviceToHost);
  cudaFree(input_d);
  cudaFree(output_d);
  cudaFree(hidden_d);
  cudaFree(cell_d);
  cudaFree(w_ih_d);
  cudaFree(w_hh_d);
  cudaFree(b_ih_d);
  cudaFree(b_hh_d);
  cudaFree(w_ho_d);
  return 0;
}

__global__ void batch_matmul_64_2048_1_512_kernel(void *__restrict__ A,
                                                  void *__restrict__ B,
                                                  void *__restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[512];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[((((int)threadIdx.y) * 64))] =
        ((float *)
             A)[((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                  (k_outer * 64)))];
    A_shared[(((((int)threadIdx.y) * 64) + 1))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  1))];
    A_shared[(((((int)threadIdx.y) * 64) + 2))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  2))];
    A_shared[(((((int)threadIdx.y) * 64) + 3))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  3))];
    A_shared[(((((int)threadIdx.y) * 64) + 4))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  4))];
    A_shared[(((((int)threadIdx.y) * 64) + 5))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  5))];
    A_shared[(((((int)threadIdx.y) * 64) + 6))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  6))];
    A_shared[(((((int)threadIdx.y) * 64) + 7))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  7))];
    A_shared[(((((int)threadIdx.y) * 64) + 8))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  8))];
    A_shared[(((((int)threadIdx.y) * 64) + 9))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  9))];
    A_shared[(((((int)threadIdx.y) * 64) + 10))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  10))];
    A_shared[(((((int)threadIdx.y) * 64) + 11))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  11))];
    A_shared[(((((int)threadIdx.y) * 64) + 12))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  12))];
    A_shared[(((((int)threadIdx.y) * 64) + 13))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  13))];
    A_shared[(((((int)threadIdx.y) * 64) + 14))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  14))];
    A_shared[(((((int)threadIdx.y) * 64) + 15))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  15))];
    A_shared[(((((int)threadIdx.y) * 64) + 16))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  16))];
    A_shared[(((((int)threadIdx.y) * 64) + 17))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  17))];
    A_shared[(((((int)threadIdx.y) * 64) + 18))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  18))];
    A_shared[(((((int)threadIdx.y) * 64) + 19))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  19))];
    A_shared[(((((int)threadIdx.y) * 64) + 20))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  20))];
    A_shared[(((((int)threadIdx.y) * 64) + 21))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  21))];
    A_shared[(((((int)threadIdx.y) * 64) + 22))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  22))];
    A_shared[(((((int)threadIdx.y) * 64) + 23))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  23))];
    A_shared[(((((int)threadIdx.y) * 64) + 24))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  24))];
    A_shared[(((((int)threadIdx.y) * 64) + 25))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  25))];
    A_shared[(((((int)threadIdx.y) * 64) + 26))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  26))];
    A_shared[(((((int)threadIdx.y) * 64) + 27))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  27))];
    A_shared[(((((int)threadIdx.y) * 64) + 28))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  28))];
    A_shared[(((((int)threadIdx.y) * 64) + 29))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  29))];
    A_shared[(((((int)threadIdx.y) * 64) + 30))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  30))];
    A_shared[(((((int)threadIdx.y) * 64) + 31))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  31))];
    A_shared[(((((int)threadIdx.y) * 64) + 32))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  32))];
    A_shared[(((((int)threadIdx.y) * 64) + 33))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  33))];
    A_shared[(((((int)threadIdx.y) * 64) + 34))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  34))];
    A_shared[(((((int)threadIdx.y) * 64) + 35))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  35))];
    A_shared[(((((int)threadIdx.y) * 64) + 36))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  36))];
    A_shared[(((((int)threadIdx.y) * 64) + 37))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  37))];
    A_shared[(((((int)threadIdx.y) * 64) + 38))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  38))];
    A_shared[(((((int)threadIdx.y) * 64) + 39))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  39))];
    A_shared[(((((int)threadIdx.y) * 64) + 40))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  40))];
    A_shared[(((((int)threadIdx.y) * 64) + 41))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  41))];
    A_shared[(((((int)threadIdx.y) * 64) + 42))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  42))];
    A_shared[(((((int)threadIdx.y) * 64) + 43))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  43))];
    A_shared[(((((int)threadIdx.y) * 64) + 44))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  44))];
    A_shared[(((((int)threadIdx.y) * 64) + 45))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  45))];
    A_shared[(((((int)threadIdx.y) * 64) + 46))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  46))];
    A_shared[(((((int)threadIdx.y) * 64) + 47))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  47))];
    A_shared[(((((int)threadIdx.y) * 64) + 48))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  48))];
    A_shared[(((((int)threadIdx.y) * 64) + 49))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  49))];
    A_shared[(((((int)threadIdx.y) * 64) + 50))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  50))];
    A_shared[(((((int)threadIdx.y) * 64) + 51))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  51))];
    A_shared[(((((int)threadIdx.y) * 64) + 52))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  52))];
    A_shared[(((((int)threadIdx.y) * 64) + 53))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  53))];
    A_shared[(((((int)threadIdx.y) * 64) + 54))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  54))];
    A_shared[(((((int)threadIdx.y) * 64) + 55))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  55))];
    A_shared[(((((int)threadIdx.y) * 64) + 56))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  56))];
    A_shared[(((((int)threadIdx.y) * 64) + 57))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  57))];
    A_shared[(((((int)threadIdx.y) * 64) + 58))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  58))];
    A_shared[(((((int)threadIdx.y) * 64) + 59))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  59))];
    A_shared[(((((int)threadIdx.y) * 64) + 60))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  60))];
    A_shared[(((((int)threadIdx.y) * 64) + 61))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  61))];
    A_shared[(((((int)threadIdx.y) * 64) + 62))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  62))];
    A_shared[(((((int)threadIdx.y) * 64) + 63))] =
        ((float *)
             A)[(((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) +
                   (k_outer * 64)) +
                  63))];
    if (((int)threadIdx.y) < 1) {
      B_shared[((((int)threadIdx.y) * 64))] =
          ((float *)B)[(((((int)threadIdx.y) * 2048) + (k_outer * 64)))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 1))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 1))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 2))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 2))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 3))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 3))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 4))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 4))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 5))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 5))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 6))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 6))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 7))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 7))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 8))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 8))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 9))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 9))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 10))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 10))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 11))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 11))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 12))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 12))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 13))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 13))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 14))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 14))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 15))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 15))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 16))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 16))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 17))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 17))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 18))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 18))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 19))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 19))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 20))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 20))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 21))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 21))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 22))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 22))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 23))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 23))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 24))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 24))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 25))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 25))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 26))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 26))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 27))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 27))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 28))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 28))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 29))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 29))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 30))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 30))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 31))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 31))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 32))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 32))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 33))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 33))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 34))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 34))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 35))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 35))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 36))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 36))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 37))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 37))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 38))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 38))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 39))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 39))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 40))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 40))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 41))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 41))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 42))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 42))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 43))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 43))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 44))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 44))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 45))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 45))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 46))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 46))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 47))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 47))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 48))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 48))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 49))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 49))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 50))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 50))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 51))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 51))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 52))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 52))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 53))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 53))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 54))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 54))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 55))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 55))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 56))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 56))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 57))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 57))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 58))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 58))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 59))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 59))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 60))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 60))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 61))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 61))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 62))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 62))];
    }
    if (((int)threadIdx.y) < 1) {
      B_shared[(((((int)threadIdx.y) * 64) + 63))] =
          ((float *)B)[((((((int)threadIdx.y) * 2048) + (k_outer * 64)) + 63))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 64; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 64) + k_inner))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.y) * 8) + ((int)threadIdx.y)))] =
      compute_local[(0)];
}

__global__ void batch_matmul_1_256_1_512_kernel(void *__restrict__ A,
                                                void *__restrict__ B,
                                                void *__restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[512];
  __shared__ float B_shared[8];
  float A_shared_local[2];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    compute_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
#pragma unroll
    for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
#pragma unroll
      for (int ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
        A_shared[(
            (((((int)threadIdx.y) * 16) + (ax1_inner * 8)) + ax2_inner))] =
            ((float *)A)[(
                (((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 1024)) +
                   (ax1_inner * 512)) +
                  (k_outer * 8)) +
                 ax2_inner))];
      }
    }
#pragma unroll
    for (int ax2_inner1 = 0; ax2_inner1 < 8; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 8) + ax2_inner1))] = ((float *)B)[(
            (((((int)threadIdx.y) * 512) + (k_outer * 8)) + ax2_inner1))];
      }
    }
    __syncthreads();
#pragma unroll
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
#pragma unroll
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        A_shared_local[(ax1)] =
            A_shared[((((((int)threadIdx.y) * 16) + (ax1 * 8)) + k_inner))];
      }
      B_shared_local[(0)] = B_shared[(k_inner)];
#pragma unroll
      for (int i_c = 0; i_c < 2; ++i_c) {
        compute_local[(i_c)] = (compute_local[(i_c)] +
                                (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
#pragma unroll
  for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
    ((float *)compute)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 2)) +
                         i_inner_inner))] = compute_local[(i_inner_inner)];
  }
}

__global__ void batch_matmul_1_2048_1_256_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[512];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 4; ++k_outer) {
    __syncthreads();
    for (int ax2_inner = 0; ax2_inner < 64; ++ax2_inner) {
      A_shared[(((((int)threadIdx.y) * 64) + ax2_inner))] =
          ((float *)
               A)[(((((((int)blockIdx.y) * 2048) + (((int)threadIdx.y) * 256)) +
                     (k_outer * 64)) +
                    ax2_inner))];
    }
    for (int ax2_inner1 = 0; ax2_inner1 < 64; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 64) + ax2_inner1))] = ((float *)B)[(
            (((((int)threadIdx.y) * 256) + (k_outer * 64)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 64; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 64) + k_inner))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.y) * 8) + ((int)threadIdx.y)))] =
      compute_local[(0)];
}

__global__ void batch_matmul_1_2048_1_512_kernel(void *__restrict__ A,
                                                 void *__restrict__ B,
                                                 void *__restrict__ compute) {
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
        A_shared[(
            (((((int)threadIdx.y) * 128) + (ax1_inner * 32)) + ax2_inner))] =
            ((float *)A)[(
                (((((((int)blockIdx.y) * 4096) + (((int)threadIdx.y) * 2048)) +
                   (ax1_inner * 512)) +
                  (k_outer * 32)) +
                 ax2_inner))];
      }
    }
    for (int ax2_inner1 = 0; ax2_inner1 < 32; ++ax2_inner1) {
      if (((int)threadIdx.y) < 1) {
        B_shared[(((((int)threadIdx.y) * 32) + ax2_inner1))] = ((float *)B)[(
            (((((int)threadIdx.y) * 512) + (k_outer * 32)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      A_shared_local[(0)] = A_shared[(((((int)threadIdx.y) * 128) + k_inner))];
      A_shared_local[(1)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 32))];
      A_shared_local[(2)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 64))];
      A_shared_local[(3)] =
          A_shared[((((((int)threadIdx.y) * 128) + k_inner) + 96))];
      B_shared_local[(0)] = B_shared[(k_inner)];
      compute_local[(0)] =
          (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] =
          (compute_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(2)] =
          (compute_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(3)] =
          (compute_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
    }
  }
  ((float *)compute)[(((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)))] =
      compute_local[(0)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 1))] =
      compute_local[(1)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 2))] =
      compute_local[(2)];
  ((float *)
       compute)[((((((int)blockIdx.y) * 8) + (((int)threadIdx.y) * 4)) + 3))] =
      compute_local[(3)];
}
