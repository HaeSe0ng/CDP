#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <time.h>

#include "seq2seq.h"
#include "util.h"

int main(int argc, char **argv)
{

  int batch_size = 4;
  int emb_dim = 512; // embedding_dim
  int seq_length = 15;
  int hidden_size = 512;
  int src_vocab_size = 16384;
  int tgt_vocab_size = 16384;
  int max_len = 400;

  int64_t sos = 0;
  int64_t *input, *output;
  set_dis_rng(0, (int64_t)src_vocab_size - 1);
  alloc_rand_mat<int64_t>(&input, seq_length, batch_size);
  /*
  for (int i = 0; i < seq_length; i++)
  {
    for (int b = 0; b < batch_size; b++)
    {
      printf("%ld ", input[i * batch_size + b]);
    }
    printf("\n");
  }*/
  // alloc_mat(&output, batch_size, tgt_vocab_size * TEMP_OUTPUT_SEQ_LENGTH);

  int64_t *eos;
  alloc_mat<int64_t>(&eos, 1, batch_size);

  ifstream out;
  char out_fname[20];
  sprintf(out_fname, "out_%d.dat", batch_size);
  out.open(out_fname, ios::binary | ios::in);

  for (int b = 0; b < batch_size; b++)
  {
    out.read(reinterpret_cast<char *>(&eos[b]), sizeof(int64_t));
  }

  out.close();
  /*
  for (int b = 0; b < batch_size; b++) {
    printf("b=%d, eos=%d\n", b, eos[b]);
  }*/

  seq2seq_inf(input, output, sos, eos, emb_dim, seq_length, hidden_size,
              batch_size, src_vocab_size, tgt_vocab_size, max_len);
  free(input);
  free(eos);
  return 0;
}