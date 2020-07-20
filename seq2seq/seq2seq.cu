#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdlib>

#include "seq2seq.h"
#include "util.h"

template <typename T>
__device__ __forceinline__ T sigmoid(T in) {
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
    dim3 gridDim(32, 128);
    dim3 blockDim(16, 4);
    matmul_kernel<<<gridDim, blockDim>>>(w_ih_d, input_d + input_dim * i,
                                         igate_d, (4 * totalElements), 1,
                                         totalInputs);
    dim3 gridDim2(32, 128);
    dim3 blockDim2(16, 4);
    matmul_kernel<<<gridDim2, blockDim2>>>(
        w_hh_d, hidden_d, hgate_d, (4 * totalElements), 1, totalElements);
    lstm_cell_kernel<<<10, 512>>>(igate_d, hgate_d, b_ih_d, b_hh_d, cell_d,
                                  hidden_d, cell_d, hidden_size, totalElements);
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
    dim3 gridDim(32, 128);
    dim3 blockDim(16, 4);
    matmul_kernel<<<gridDim, blockDim>>>(w_ih_d, input_d + input_dim * i,
                                         igate_d, (4 * totalElements), 1,
                                         totalInputs);
    dim3 gridDim2(32, 128);
    dim3 blockDim2(16, 4);
    matmul_kernel<<<gridDim2, blockDim2>>>(
        w_hh_d, hidden_d, hgate_d, (4 * totalElements), 1, totalElements);
    lstm_cell_kernel<<<10, 512>>>(igate_d, hgate_d, b_ih_d, b_hh_d, cell_d,
                                  hidden_d, cell_d, hidden_size, totalElements);
    dim3 gridDim3(32, 128);
    dim3 blockDim3(16, 4);
    matmul_kernel<<<gridDim3, blockDim3>>>(w_ho_d, hidden_d,
                                           output_d + input_dim * i,
                                           totalInputs, 1, totalElements);
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

  cudaMemcpy(output, output_d, (sizeof(float) * totalElements),
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
