#include "tvm_kernels.cuh"
#include <cuda_device_runtime_api.h>

#include <stdio.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "seq2seq.h"
#include "util.h"

#define AT_APPLY_THREADS_PER_BLOCK 512

__device__ int out_seq_len_d = 20;

void batch_matmul(float *A, float *B, float *C, int bsz, int M, int N, int K);
__device__ void batch_matmul_dev(float *A, float *B, float *C, int bsz, int M,
                                 int N, int K);

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
__global__ void argmax_kernel(float *input_d, int *output_d, int bsz,
                              int input_len) {
  float temp_topv, temp_v;
  int temp_topi;
  for (int b = 0; b < bsz; b++) {
    temp_topv = 0;
    temp_topi = 0;
    for (int vocab_idx = 0; vocab_idx < input_len; vocab_idx++) {
      temp_v = input_d[b * input_len + vocab_idx];
      if (temp_v > temp_topv) {
        temp_topi = vocab_idx;
        temp_topv = temp_v;
      }
    }
    output_d[b] = temp_topi;
    // printf("argmax] b=%d,temp_topi=%d,temp_topv=%f\n", b, temp_topi,
    // temp_topv);
  }
}
void lstm(float *input_d, float *hidden_d, float *w_ih_d, float *w_hh_d,
          float *igate_d, float *hgate_d, float *b_ih_d, float *b_hh_d,
          float *cell_d, int bsz, int input_dim, int hidden_size,
          int totalElements) {
  batch_matmul(input_d, w_ih_d, igate_d, 1, bsz, 4 * hidden_size,
               input_dim); // bsz, 4*hidden_size, input_dim
  batch_matmul(hidden_d, w_hh_d, hgate_d, 1, bsz, 4 * hidden_size,
               hidden_size); // bsz, 4*hidden_size, hidden_size
  lstm_cell_kernel<<<totalElements / AT_APPLY_THREADS_PER_BLOCK,
                     AT_APPLY_THREADS_PER_BLOCK>>>(
      igate_d, hgate_d, b_ih_d, b_hh_d, cell_d, hidden_d, cell_d, hidden_size,
      totalElements);
}
__device__ void lstm_dev(float *input_d, float *hidden_d, float *w_ih_d,
                         float *w_hh_d, float *igate_d, float *hgate_d,
                         float *b_ih_d, float *b_hh_d, float *cell_d, int bsz,
                         int input_dim, int hidden_size, int totalElements) {
  batch_matmul_dev(input_d, w_ih_d, igate_d, 1, bsz, 4 * hidden_size,
                   input_dim); // bsz, 4*hidden_size, input_dim
  batch_matmul_dev(hidden_d, w_hh_d, hgate_d, 1, bsz, 4 * hidden_size,
                   hidden_size); // bsz, 4*hidden_size, hidden_size
  lstm_cell_kernel<<<totalElements / AT_APPLY_THREADS_PER_BLOCK,
                     AT_APPLY_THREADS_PER_BLOCK>>>(
      igate_d, hgate_d, b_ih_d, b_hh_d, cell_d, hidden_d, cell_d, hidden_size,
      totalElements);
}
void embedding(int *input, int emb_dim, int bsz, float *emb_tbl_d,
               float *emb_vec_d) {
  for (int b = 0; b < bsz; b++) {
    cudaMemcpy(emb_vec_d + b * emb_dim, emb_tbl_d + input[b] * emb_dim,
               (sizeof(float) * emb_dim), cudaMemcpyDeviceToDevice);
  }
}
__device__ void embedding_dev(int *input_d, int emb_dim, int bsz,
                              float *emb_tbl_d, float *emb_vec_d) {
  for (int b = 0; b < bsz; b++) {
    memcpy(emb_vec_d + b * emb_dim, emb_tbl_d + input_d[b] * emb_dim,
           (sizeof(float) * emb_dim));
  }
}
__device__ void argmax_dev(float *input, int *output, int bsz, int input_len) {
  argmax_kernel<<<1, 1>>>(input, output, bsz, input_len);
}
void seq2seq_encode(int *input, float *emb_tbl_d, float *emb_vec_d,
                    float *hidden_d, float *w_ih_d, float *w_hh_d,
                    float *igate_d, float *hgate_d, float *b_ih_d,
                    float *b_hh_d, float *cell_d, float *w_ho_d, int bsz,
                    int emb_dim, int hidden_size, int totalElements,
                    int seq_length) {
  for (int i = 0; i < seq_length; i++) {
    embedding(input + i * bsz, emb_dim, bsz, emb_tbl_d, emb_vec_d);

    lstm(emb_vec_d, hidden_d, w_ih_d, w_hh_d, igate_d, hgate_d, b_ih_d, b_hh_d,
         cell_d, bsz, emb_dim, hidden_size, totalElements);
  }
}

__global__ void seq2seq_decode(
    float *emb_tbl_d, float *emb_vec_d, float *hidden_d, float *w_ih_d,
    float *w_hh_d, float *igate_d, float *hgate_d, float *b_ih_d, float *b_hh_d,
    float *cell_d, float *output_onehot_d, float *w_ho_d, int *output_d,
    int *output, int *eos_d, int bsz, int emb_dim, int hidden_size,
    int totalElements, int tgt_vocab_size, int max_len, int *sos_batch_d) {
  int i;
  bool is_end;
  for (i = 0; i < max_len; i++) {
    is_end = true;
    if (i == 0)
      embedding_dev(sos_batch_d, emb_dim, bsz, emb_tbl_d, emb_vec_d);
    else
      embedding_dev(output_d + bsz * (i - 1), emb_dim, bsz, emb_tbl_d,
                    emb_vec_d);
    lstm_dev(emb_vec_d, hidden_d, w_ih_d, w_hh_d, igate_d, hgate_d, b_ih_d,
             b_hh_d, cell_d, bsz, emb_dim, hidden_size, totalElements);
    batch_matmul_dev(hidden_d, w_ho_d,
                     output_onehot_d + bsz * tgt_vocab_size * i, 1, bsz,
                     tgt_vocab_size,
                     hidden_size); // bsz, tgt_vocab_size, hidden_size
    argmax_dev(output_onehot_d + bsz * tgt_vocab_size * i, output_d + bsz * i,
               bsz, tgt_vocab_size);
    cudaDeviceSynchronize();
    //__syncthreads();
    for (int b = 0; b < bsz; b++) {
      // printf("i=%d, output_d[%d]=%d, eos_d[%d]=%d\n", i, bsz * i + b,
      //       output_d[bsz * i + b], b, eos_d[b]);
      if (output_d[bsz * i + b] != eos_d[b]) {
        is_end = false;
        break;
      }
    }
    if (is_end) {
      printf("end: out_seq_len=%d\n", i + 1);
      out_seq_len_d = i + 1;
      break;
    }
  }
}
int seq2seq_inf(int *input, int *output, int sos, int *eos, int emb_dim,
                int seq_length, int hidden_size, int batch_size,
                int src_vocab_size, int tgt_vocab_size, int max_len) {
  cudaMallocHost((void **)&output, sizeof(int) * batch_size * max_len);
  int *sos_batch_d;
  cudaMalloc((void **)&sos_batch_d, sizeof(int) * batch_size);
  cudaMemset(sos_batch_d, sos, sizeof(int) * batch_size);

  int totalElements = batch_size * hidden_size;
  int *output_d, *eos_d;
  float *w_ih_enc, *w_hh_enc, *b_ih_enc, *b_hh_enc;
  float *w_ih_dec, *w_hh_dec, *b_ih_dec, *b_hh_dec, *w_ho;
  float *emb_tbl_enc, *emb_tbl_dec;
  float *hidden_d, *igate_d, *hgate_d, *cell_d;
  float *w_ih_enc_d, *w_hh_enc_d, *b_ih_enc_d, *b_hh_enc_d;
  float *w_ih_dec_d, *w_hh_dec_d, *b_ih_dec_d, *b_hh_dec_d, *w_ho_d;
  float *output_onehot_d, *emb_tbl_enc_d, *emb_tbl_dec_d, *emb_vec_d;

  cudaMalloc((void **)&eos_d, sizeof(int) * batch_size);
  cudaMemcpy(eos_d, eos, (sizeof(int) * batch_size), cudaMemcpyHostToDevice);

  alloc_rand_mat<float>(&emb_tbl_enc, src_vocab_size, emb_dim);
  cudaMalloc((void **)&emb_tbl_enc_d, sizeof(float) * src_vocab_size * emb_dim);
  alloc_rand_mat<float>(&emb_tbl_dec, tgt_vocab_size, emb_dim);
  cudaMalloc((void **)&emb_tbl_dec_d, sizeof(float) * tgt_vocab_size * emb_dim);
  cudaMalloc((void **)&emb_vec_d, sizeof(float) * batch_size * emb_dim);

  cudaMemcpy(emb_tbl_enc_d, emb_tbl_enc,
             (sizeof(float) * src_vocab_size * emb_dim),
             cudaMemcpyHostToDevice);
  cudaMemcpy(emb_tbl_dec_d, emb_tbl_dec,
             (sizeof(float) * tgt_vocab_size * emb_dim),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&output_onehot_d,
             sizeof(float) * batch_size * tgt_vocab_size * max_len);
  cudaMalloc((void **)&output_d, sizeof(int) * batch_size * max_len);

  cudaMalloc((void **)&hidden_d, sizeof(float) * batch_size * hidden_size);
  cudaMalloc((void **)&cell_d, sizeof(float) * batch_size * hidden_size);

  cudaMalloc((void **)&igate_d, sizeof(float) * batch_size * (4 * hidden_size));
  cudaMalloc((void **)&hgate_d, sizeof(float) * batch_size * (4 * hidden_size));

  cudaMalloc((void **)&w_ih_enc_d, sizeof(float) * (4 * hidden_size) * emb_dim);
  cudaMalloc((void **)&w_hh_enc_d,
             sizeof(float) * (4 * hidden_size) * hidden_size);
  cudaMalloc((void **)&b_ih_enc_d, sizeof(float) * 4 * hidden_size);
  cudaMalloc((void **)&b_hh_enc_d, sizeof(float) * 4 * hidden_size);

  cudaMalloc((void **)&w_ih_dec_d, sizeof(float) * (4 * hidden_size) * emb_dim);
  cudaMalloc((void **)&w_hh_dec_d,
             sizeof(float) * (4 * hidden_size) * hidden_size);
  cudaMalloc((void **)&b_ih_dec_d, sizeof(float) * 4 * hidden_size);
  cudaMalloc((void **)&b_hh_dec_d, sizeof(float) * 4 * hidden_size);
  cudaMalloc((void **)&w_ho_d, sizeof(float) * tgt_vocab_size * hidden_size);

  cudaMemset(hidden_d, 0, sizeof(float) * batch_size * hidden_size);
  cudaMemset(cell_d, 0, sizeof(float) * batch_size * hidden_size);
  alloc_rand_mat<float>(&w_ih_enc, (4 * hidden_size), emb_dim);
  alloc_rand_mat<float>(&w_hh_enc, (4 * hidden_size), hidden_size);
  alloc_rand_mat<float>(&b_ih_enc, 1, 4 * hidden_size);
  alloc_rand_mat<float>(&b_hh_enc, 1, 4 * hidden_size);

  cudaMemcpy(w_ih_enc_d, w_ih_enc,
             (sizeof(float) * emb_dim * (4 * hidden_size)),
             cudaMemcpyHostToDevice);
  cudaMemcpy(w_hh_enc_d, w_hh_enc,
             (sizeof(float) * hidden_size * (4 * hidden_size)),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_ih_enc_d, b_ih_enc, (sizeof(float) * 4 * hidden_size),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_hh_enc_d, b_hh_enc, (sizeof(float) * 4 * hidden_size),
             cudaMemcpyHostToDevice);

  alloc_rand_mat<float>(&w_ih_dec, (4 * hidden_size), emb_dim);
  alloc_rand_mat<float>(&w_hh_dec, (4 * hidden_size), hidden_size);
  alloc_rand_mat<float>(&b_ih_dec, 1, 4 * hidden_size);
  alloc_rand_mat<float>(&b_hh_dec, 1, 4 * hidden_size);
  cudaMemcpy(w_ih_dec_d, w_ih_dec,
             (sizeof(float) * emb_dim * (4 * hidden_size)),
             cudaMemcpyHostToDevice);
  cudaMemcpy(w_hh_dec_d, w_hh_dec,
             (sizeof(float) * hidden_size * (4 * hidden_size)),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_ih_dec_d, b_ih_dec, (sizeof(float) * 4 * hidden_size),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_hh_dec_d, b_hh_dec, (sizeof(float) * 4 * hidden_size),
             cudaMemcpyHostToDevice);

  alloc_rand_mat<float>(&w_ho, tgt_vocab_size, hidden_size);
  cudaMemcpy(w_ho_d, w_ho, (sizeof(float) * hidden_size * tgt_vocab_size),
             cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // main logic

  seq2seq_encode(input, emb_tbl_enc_d, emb_vec_d, hidden_d, w_ih_enc_d,
                 w_hh_enc_d, igate_d, hgate_d, b_ih_enc_d, b_hh_enc_d, cell_d,
                 w_ho_d, batch_size, emb_dim, hidden_size, totalElements,
                 seq_length);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("[CDP_encode]execution time: %fms\n", elapsed_time);
  cudaEventRecord(start);
  int out_seq_len;
  seq2seq_decode<<<1, 1>>>(emb_tbl_dec_d, emb_vec_d, hidden_d, w_ih_dec_d,
                           w_hh_dec_d, igate_d, hgate_d, b_ih_dec_d, b_hh_dec_d,
                           cell_d, output_onehot_d, w_ho_d, output_d, output,
                           eos_d, batch_size, emb_dim, hidden_size,
                           totalElements, tgt_vocab_size, max_len, sos_batch_d);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("[CDP_decode]execution time: %fms\n", elapsed_time);

  cudaEventRecord(start);

  cudaMemcpyFromSymbol(&out_seq_len, out_seq_len_d, sizeof(out_seq_len), 0,
                       cudaMemcpyDeviceToHost);
  cudaMemcpy(output, output_d, (sizeof(int) * batch_size * out_seq_len),
             cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("[CDP_decode_memcpy]execution time: %fms\n", elapsed_time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(emb_tbl_enc_d);
  cudaFree(emb_tbl_dec_d);
  cudaFree(output_onehot_d);
  cudaFree(output_d);
  cudaFree(hidden_d);
  cudaFree(cell_d);
  cudaFree(igate_d);
  cudaFree(hgate_d);
  cudaFree(w_ih_enc_d);
  cudaFree(w_hh_enc_d);
  cudaFree(b_ih_enc_d);
  cudaFree(b_hh_enc_d);
  cudaFree(w_ih_dec_d);
  cudaFree(w_hh_dec_d);
  cudaFree(b_ih_dec_d);
  cudaFree(b_hh_dec_d);
  cudaFree(w_ho_d);

  cudaFreeHost(output);
  free(w_ih_enc);
  free(w_hh_enc);
  free(b_ih_enc);
  free(b_hh_enc);
  free(w_ih_dec);
  free(w_hh_dec);
  free(b_ih_dec);
  free(b_hh_dec);
  free(w_ho);
  free(emb_tbl_enc);
  free(emb_tbl_dec);

  return 0;
}

void batch_matmul(float *A, float *B, float *C, int bsz, int M, int N, int K) {
  if (bsz == 1 && M == 2048 && N == 1 && K == 512) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 2, 1);
    batch_matmul_1_2048_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 2048 && N == 1 && K == 256) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 8, 1);
    batch_matmul_1_2048_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 16384 && N == 1 && K == 512) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 16, 1);
    batch_matmul_1_16384_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 256 && N == 1 && K == 512) {
    dim3 gridDim(1, 4, 1);
    dim3 blockDim(1, 32, 1);
    batch_matmul_1_256_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 64 && M == 2048 && N == 1 && K == 512) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 2, 1);
    batch_matmul_64_2048_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 1 && N == 2048 && K == 512) {
    dim3 gridDim(64, 1, 1);
    dim3 blockDim(8, 1, 1);
    batch_matmul_1_1_2048_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 1 && N == 16384 && K == 512) {
    dim3 gridDim(512, 1, 1);
    dim3 blockDim(32, 1, 1);
    batch_matmul_1_1_16384_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 4 && N == 2048 && K == 512) {
    dim3 gridDim(256, 1, 1);
    dim3 blockDim(8, 4, 1);
    batch_matmul_1_4_2048_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 4 && N == 16384 && K == 512) {
    dim3 gridDim(256, 1, 1);
    dim3 blockDim(32, 1, 1);
    batch_matmul_1_4_16384_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 64 && N == 2048 && K == 512) {
    dim3 gridDim(64, 2, 1);
    dim3 blockDim(16, 2, 1);
    batch_matmul_1_64_2048_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 64 && N == 16384 && K == 512) {
    dim3 gridDim(256, 1, 1);
    dim3 blockDim(32, 2, 1);
    batch_matmul_1_64_16384_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else {
    printf("batch_matmul: WRONG ARGS (bsz=%d, M=%d, N=%d, K=%d)", bsz, M, N, K);
    exit(-1);
  }
}
__device__ void batch_matmul_dev(float *A, float *B, float *C, int bsz, int M,
                                 int N, int K) {
  if (bsz == 1 && M == 2048 && N == 1 && K == 512) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 2, 1);
    batch_matmul_1_2048_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 2048 && N == 1 && K == 256) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 8, 1);
    batch_matmul_1_2048_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 16384 && N == 1 && K == 512) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 16, 1);
    batch_matmul_1_16384_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 256 && N == 1 && K == 512) {
    dim3 gridDim(1, 4, 1);
    dim3 blockDim(1, 32, 1);
    batch_matmul_1_256_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 64 && M == 2048 && N == 1 && K == 512) {
    dim3 gridDim(1, 256, 1);
    dim3 blockDim(1, 2, 1);
    batch_matmul_64_2048_1_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 1 && N == 2048 && K == 512) {
    dim3 gridDim(64, 1, 1);
    dim3 blockDim(8, 1, 1);
    batch_matmul_1_1_2048_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 1 && N == 16384 && K == 512) {
    dim3 gridDim(512, 1, 1);
    dim3 blockDim(32, 1, 1);
    batch_matmul_1_1_16384_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 4 && N == 2048 && K == 512) {
    dim3 gridDim(256, 1, 1);
    dim3 blockDim(8, 4, 1);
    batch_matmul_1_4_2048_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 4 && N == 16384 && K == 512) {
    dim3 gridDim(256, 1, 1);
    dim3 blockDim(32, 1, 1);
    batch_matmul_1_4_16384_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 64 && N == 2048 && K == 512) {
    dim3 gridDim(64, 2, 1);
    dim3 blockDim(16, 2, 1);
    batch_matmul_1_64_2048_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else if (bsz == 1 && M == 64 && N == 16384 && K == 512) {
    dim3 gridDim(256, 1, 1);
    dim3 blockDim(32, 2, 1);
    batch_matmul_1_64_16384_512_kernel<<<gridDim, blockDim>>>(A, B, C);
  } else {
    printf("batch_matmul: WRONG ARGS (bsz=%d, M=%d, N=%d, K=%d)", bsz, M, N, K);
  }
}