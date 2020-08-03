#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "reduce.cuh"
#include "seq2seq.h"
#include "torch_kernels.cuh"
#include "tvm_kernels.cuh"
#include "util.h"

#define TEMP_OUTPUT_SEQ_LENGTH 18

void embedding(int64_t *input_d, int emb_dim, int bsz, float *emb_tbl_d,
               float *emb_vec_d) {
  ptrdiff_t numIndices = bsz;

  ptrdiff_t outTotalSize = bsz * emb_dim;
  if (outTotalSize == 0) {
    return;
  }

  ptrdiff_t sliceSize = outTotalSize / numIndices;
  /*
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  */
  int mpc = 30; // prop.multiProcessorCount;

  // A reasonable choice for when to have each thread iterate over
  // indices to choose
  if (numIndices <= 16) {
    dim3 smallIndexGrid(
        MIN(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
    dim3 smallIndexBlock(MIN(sliceSize, (ptrdiff_t)128));
    indexSelectSmallIndex<<<smallIndexGrid, smallIndexBlock, 0>>>(
        emb_vec_d, emb_tbl_d, input_d, sliceSize, numIndices);
  } else {
    dim3 largeIndexGrid(
        MIN(THCCeilDiv(outTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
    dim3 largeIndexBlock(MIN(outTotalSize, (ptrdiff_t)128));
    indexSelectLargeIndex<<<largeIndexGrid, largeIndexBlock, 0>>>(
        emb_vec_d, emb_tbl_d, input_d, outTotalSize, sliceSize);
  }
}
void argmax(float *input_d, int64_t *output_d, int bsz, int input_len) {
  int num_outputs = bsz;
  int inputs_per_output = input_len;

  // using traits = function_traits<decltype(&ArgMaxOps<float>::reduce)>;
  // using arg_t = typename decay<typename traits::template arg<0>::type>::type;
  static constexpr int vt0 = 4;
  auto ident =
      thrust::pair<float, int64_t>(at::numeric_limits<float>::lower_bound(), 0);

  auto config = ReduceConfig(16, num_outputs, inputs_per_output);
  config.set_block_dimension(inputs_per_output, num_outputs);
  int block_width = config.block_width;
  int block_height = config.block_height;
  config.input_mult[0] = config.split_input(block_width);
  if (config.values_per_thread() >= block_height * 16 ||
      config.values_per_thread() >= 256) {
    config.input_mult[1] = config.split_input(block_height);
  } else {
    config.output_mult[1] = config.split_output(block_height);
  }

  int num_reduce_dims = 1;
  int num_output_dims = 1;
  int64_t output_strides[2] = {0, sizeof(int64_t)};
  int64_t input_strides[2] = {sizeof(float),
                              (int64_t)sizeof(float) * inputs_per_output};
  int64_t *output_calc_strides[2] = {
      output_strides + num_reduce_dims,
      input_strides + num_reduce_dims,
  };
  int64_t *input_calc_strides[1] = {
      input_strides,
  };
  int64_t shape[2] = {inputs_per_output, num_outputs};
  auto output_calc = OffsetCalculator<2, uint32_t>(
      num_output_dims, shape + num_reduce_dims, &output_calc_strides[0]);
  auto input_calc = OffsetCalculator<1, uint32_t>(num_reduce_dims, shape,
                                                  &input_calc_strides[0]);

  auto reduce = ReduceOp<float, ArgMaxOps<float>, uint32_t, int64_t, vt0>(
      ArgMaxOps<float>{}, config, input_calc, output_calc,
      (const void *)input_d, (char *)output_d, nullptr, nullptr, nullptr, ident,
      1);
  reduce.accumulate = false;
  reduce.final_output = true;

  launch_reduce_kernel<ReduceConfig::MAX_NUM_THREADS>(config, reduce);
  /*
cudaError_t err = cudaGetLastError();
if (cudaSuccess != err)
  printf("*cudaErr(%d) : %s \n", err, cudaGetErrorString(err));*/
}

void batch_matmul(float *A, float *B, float *C, int bsz, int M, int N, int K);

__global__ void argmax_naive_kernel(float *input_d, int64_t *output_d, int bsz,
                                    int64_t input_len) {
  float temp_topv, temp_v;
  int64_t temp_topi;
  for (int b = 0; b < bsz; b++) {
    temp_topv = 0;
    temp_topi = 0;
    for (int64_t vocab_idx = 0; vocab_idx < input_len; vocab_idx++) {
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

void argmax_naive(float *input_d, int64_t *output_d, int bsz, int input_len) {
  argmax_naive_kernel<<<1, 1>>>(input_d, output_d, bsz, (int64_t)input_len);
}
void seq2seq_encode(int64_t *input_d, float *emb_tbl_d, float *emb_vec_d,
                    float *hidden_d, float *w_ih_d, float *w_hh_d,
                    float *igate_d, float *hgate_d, float *b_ih_d,
                    float *b_hh_d, float *cell_d, float *w_ho_d, int bsz,
                    int emb_dim, int hidden_size, int totalElements,
                    int seq_length) {
  for (int i = 0; i < seq_length; i++) {
    embedding(input_d + i * bsz, emb_dim, bsz, emb_tbl_d, emb_vec_d);

    lstm(emb_vec_d, hidden_d, w_ih_d, w_hh_d, igate_d, hgate_d, b_ih_d, b_hh_d,
         cell_d, bsz, emb_dim, hidden_size, totalElements);
  }
}
void seq2seq_decode_save(float *emb_tbl_d, float *emb_vec_d, float *hidden_d,
                         float *w_ih_d, float *w_hh_d, float *igate_d,
                         float *hgate_d, float *b_ih_d, float *b_hh_d,
                         float *cell_d, float *output_onehot_d, float *w_ho_d,
                         int64_t *output_d, int64_t *output, int64_t *eos,
                         int bsz, int emb_dim, int hidden_size,
                         int totalElements, int tgt_vocab_size, int max_len,
                         int64_t *sos_batch_d, int seq_length) {
  ofstream out;
  char out_fname[100];
  sprintf(out_fname, "out_%d_%d_%d_%d.dat", bsz, seq_length, emb_dim,
          tgt_vocab_size);
  out.open(out_fname, ios::out | ios::binary);
  int i;
  for (i = 0; i < seq_length; i++) {
    if (i == 0)
      embedding(sos_batch_d, emb_dim, bsz, emb_tbl_d, emb_vec_d);
    else
      embedding(output_d + bsz * (i - 1), emb_dim, bsz, emb_tbl_d, emb_vec_d);
    lstm(emb_vec_d, hidden_d, w_ih_d, w_hh_d, igate_d, hgate_d, b_ih_d, b_hh_d,
         cell_d, bsz, emb_dim, hidden_size, totalElements);
    batch_matmul(hidden_d, w_ho_d, output_onehot_d + bsz * tgt_vocab_size * i,
                 1, bsz, tgt_vocab_size,
                 hidden_size); // bsz, tgt_vocab_size, hidden_size
    argmax_naive(output_onehot_d + bsz * tgt_vocab_size * i, output_d + bsz * i,
                 bsz, tgt_vocab_size);
    cudaMemcpy(output + bsz * i, output_d + bsz * i, (sizeof(int64_t) * bsz),
               cudaMemcpyDeviceToHost);
  }
  int64_t *output_eos = output + (i - 1) * bsz;
  for (int b = 0; b < bsz; b++) {
    printf("b=%d, eos=%ld\n", b, output_eos[b]);
  }
  out.write(reinterpret_cast<const char *>(output_eos), sizeof(int64_t) * bsz);
  out.close();
}
void seq2seq_decode(float *emb_tbl_d, float *emb_vec_d, float *hidden_d,
                    float *w_ih_d, float *w_hh_d, float *igate_d,
                    float *hgate_d, float *b_ih_d, float *b_hh_d, float *cell_d,
                    float *output_onehot_d, float *w_ho_d, int64_t *output_d,
                    int64_t *output, int64_t *eos, int bsz, int emb_dim,
                    int hidden_size, int totalElements, int64_t tgt_vocab_size,
                    int max_len, int64_t *sos_batch_d) {
  int i;
  bool is_end;
  for (i = 0; i < max_len; i++) {
    is_end = true;
    if (i == 0)
      embedding(sos_batch_d, emb_dim, bsz, emb_tbl_d, emb_vec_d);
    else
      embedding(output_d + bsz * (i - 1), emb_dim, bsz, emb_tbl_d, emb_vec_d);
    lstm(emb_vec_d, hidden_d, w_ih_d, w_hh_d, igate_d, hgate_d, b_ih_d, b_hh_d,
         cell_d, bsz, emb_dim, hidden_size, totalElements);
    batch_matmul(hidden_d, w_ho_d, output_onehot_d + bsz * tgt_vocab_size * i,
                 1, bsz, tgt_vocab_size,
                 hidden_size); // bsz, tgt_vocab_size, hidden_size

    argmax(output_onehot_d + bsz * tgt_vocab_size * i, output_d + bsz * i, bsz,
           tgt_vocab_size);
    cudaMemcpy(output + bsz * i, output_d + bsz * i, (sizeof(int64_t) * bsz),
               cudaMemcpyDeviceToHost);
    for (int b = 0; b < bsz; b++) {
      if (output[bsz * i + b] != eos[b]) {
        is_end = false;
        break;
      }
    }
    if (is_end) {
      i++;
      break;
    }
  }
  // printf("end: out_seq_len=%d\n", i);
}
void seq2seq_decode_prof(float *emb_tbl_d, float *emb_vec_d, float *hidden_d,
                         float *w_ih_d, float *w_hh_d, float *igate_d,
                         float *hgate_d, float *b_ih_d, float *b_hh_d,
                         float *cell_d, float *output_onehot_d, float *w_ho_d,
                         int64_t *output_d, int64_t *output, int64_t *eos,
                         int bsz, int emb_dim, int hidden_size,
                         int totalElements, int64_t tgt_vocab_size, int max_len,
                         int64_t *sos_batch_d, int seq_length,
                         float *elapsed_time_dec, float *elapsed_time_mem) {
  bool is_end;
  float elapsed_time;
  cudaEvent_t start, decode, memcpy;
  cudaEventCreate(&start);
  cudaEventCreate(&decode);
  cudaEventCreate(&memcpy);

  cudaEventRecord(start);

  for (int i = 0; i < seq_length; i++) {
    if (i == 0)
      embedding(sos_batch_d, emb_dim, bsz, emb_tbl_d, emb_vec_d);
    else
      embedding(output_d + bsz * (i - 1), emb_dim, bsz, emb_tbl_d, emb_vec_d);
    lstm(emb_vec_d, hidden_d, w_ih_d, w_hh_d, igate_d, hgate_d, b_ih_d, b_hh_d,
         cell_d, bsz, emb_dim, hidden_size, totalElements);
    batch_matmul(hidden_d, w_ho_d, output_onehot_d + bsz * tgt_vocab_size * i,
                 1, bsz, tgt_vocab_size,
                 hidden_size); // bsz, tgt_vocab_size, hidden_size

    argmax(output_onehot_d + bsz * tgt_vocab_size * i, output_d + bsz * i, bsz,
           tgt_vocab_size);
    cudaEventRecord(decode);
    cudaMemcpyAsync(output + bsz * i, output_d + bsz * i,
                    (sizeof(int64_t) * bsz), cudaMemcpyDeviceToHost);
    cudaEventRecord(memcpy);
    cudaEventSynchronize(memcpy);
    cudaEventElapsedTime(&elapsed_time, decode, memcpy);
    *elapsed_time_mem += elapsed_time;
    for (int b = 0; b < bsz; b++) {
      if (output[bsz * i + b] != eos[b]) {
        is_end = false;
        break;
      }
    }
    if (is_end) {
      i++;
      break;
    }
  }
  cudaEventElapsedTime(&elapsed_time, start, memcpy);
  *elapsed_time_dec += elapsed_time;
}

int seq2seq_inf(int64_t *input, int64_t *output, int64_t sos, int64_t *eos,
                int emb_dim, int seq_length, int hidden_size, int batch_size,
                int src_vocab_size, int tgt_vocab_size, int max_len,
                float *res) {
  cudaMallocHost((void **)&output, sizeof(int64_t) * batch_size * max_len);
  cudaMemset(output, 0, sizeof(int64_t) * batch_size * max_len);
  int64_t *input_d;
  cudaMalloc((void **)&input_d, sizeof(int64_t) * seq_length * batch_size);
  cudaMemcpy(input_d, input, (sizeof(int64_t) * seq_length * batch_size),
             cudaMemcpyHostToDevice);

  int64_t *sos_batch_d;
  cudaMalloc((void **)&sos_batch_d, sizeof(int64_t) * batch_size);
  cudaMemset(sos_batch_d, sos, sizeof(int64_t) * batch_size);

  int totalElements = batch_size * hidden_size;
  int64_t *output_d;
  float *w_ih_enc, *w_hh_enc, *b_ih_enc, *b_hh_enc;
  float *w_ih_dec, *w_hh_dec, *b_ih_dec, *b_hh_dec, *w_ho;
  float *emb_tbl_enc, *emb_tbl_dec;
  float *hidden_d, *igate_d, *hgate_d, *cell_d;
  float *w_ih_enc_d, *w_hh_enc_d, *b_ih_enc_d, *b_hh_enc_d;
  float *w_ih_dec_d, *w_hh_dec_d, *b_ih_dec_d, *b_hh_dec_d, *w_ho_d;
  float *output_onehot_d, *emb_tbl_enc_d, *emb_tbl_dec_d, *emb_vec_d;

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
  cudaMalloc((void **)&output_d, sizeof(int64_t) * batch_size * max_len);

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
  float elapsed_time;
  float elapsed_time_enc = 0, elapsed_time_dec = 0, elapsed_time_mem = 0;

  bool prof = true;
  if (prof) {

    int num_itr = 200;
    for (int i = -num_itr; i < num_itr; i++) {
      cudaMemset(hidden_d, 0, sizeof(float) * batch_size * hidden_size);
      cudaMemset(cell_d, 0, sizeof(float) * batch_size * hidden_size);

      if (i < 0) {
        seq2seq_encode(input_d, emb_tbl_enc_d, emb_vec_d, hidden_d, w_ih_enc_d,
                       w_hh_enc_d, igate_d, hgate_d, b_ih_enc_d, b_hh_enc_d,
                       cell_d, w_ho_d, batch_size, emb_dim, hidden_size,
                       totalElements, seq_length);
        seq2seq_decode(emb_tbl_dec_d, emb_vec_d, hidden_d, w_ih_dec_d,
                       w_hh_dec_d, igate_d, hgate_d, b_ih_dec_d, b_hh_dec_d,
                       cell_d, output_onehot_d, w_ho_d, output_d, output, eos,
                       batch_size, emb_dim, hidden_size, totalElements,
                       tgt_vocab_size, max_len, sos_batch_d);
      } else {
        cudaEventRecord(start);

        seq2seq_encode(input_d, emb_tbl_enc_d, emb_vec_d, hidden_d, w_ih_enc_d,
                       w_hh_enc_d, igate_d, hgate_d, b_ih_enc_d, b_hh_enc_d,
                       cell_d, w_ho_d, batch_size, emb_dim, hidden_size,
                       totalElements, seq_length);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);

        elapsed_time_enc += elapsed_time;

        seq2seq_decode_prof(
            emb_tbl_dec_d, emb_vec_d, hidden_d, w_ih_dec_d, w_hh_dec_d, igate_d,
            hgate_d, b_ih_dec_d, b_hh_dec_d, cell_d, output_onehot_d, w_ho_d,
            output_d, output, eos, batch_size, emb_dim, hidden_size,
            totalElements, tgt_vocab_size, max_len, sos_batch_d, seq_length,
            &elapsed_time_dec, &elapsed_time_mem);
      }
    }
    elapsed_time_enc /= num_itr;
    elapsed_time_dec /= num_itr;
    elapsed_time_mem /= num_itr;
    res[0] = elapsed_time_enc;
    res[1] = elapsed_time_dec;
    res[2] = elapsed_time_mem;

  }
  // one time execution
  else {
    cudaMemset(hidden_d, 0, sizeof(float) * batch_size * hidden_size);
    cudaMemset(cell_d, 0, sizeof(float) * batch_size * hidden_size);
    seq2seq_encode(input_d, emb_tbl_enc_d, emb_vec_d, hidden_d, w_ih_enc_d,
                   w_hh_enc_d, igate_d, hgate_d, b_ih_enc_d, b_hh_enc_d, cell_d,
                   w_ho_d, batch_size, emb_dim, hidden_size, totalElements,
                   seq_length);
    seq2seq_decode(emb_tbl_dec_d, emb_vec_d, hidden_d, w_ih_dec_d, w_hh_dec_d,
                   igate_d, hgate_d, b_ih_dec_d, b_hh_dec_d, cell_d,
                   output_onehot_d, w_ho_d, output_d, output, eos, batch_size,
                   emb_dim, hidden_size, totalElements, tgt_vocab_size, max_len,
                   sos_batch_d);
    for (int i = 0; i < seq_length; i++) {
      cudaEventRecord(start);
      cudaMemcpy(output + batch_size * i, output_d + batch_size * i,
                 (sizeof(int64_t) * batch_size), cudaMemcpyDeviceToHost);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      elapsed_time_mem += elapsed_time;
    }
    printf("memcpy: %f\n", elapsed_time_mem);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(input_d);
  cudaFree(sos_batch_d);
  cudaFreeHost(output);
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
  auto cfg = matmul_kernel_launch_cfg(bsz, M, N, K);
  (*(cfg.func))<<<cfg.gridDim, cfg.blockDim>>>(A, B, C);
}