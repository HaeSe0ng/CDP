#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <time.h>

#include "util.h"
#include "seq2seq.h"

using namespace std;
//임베딩 구현 안함
int main(int argc, char **argv)
{
  srand(0);
  int batch_size = 4;
  int input_dim = 512; //embedding_dim
  int seq_length = 10;
  int hidden_size = 512;
  int tgt_vocab_size = 16384;

  float *input, *output;
  alloc_mat(&input, batch_size, input_dim * seq_length);
  //alloc_mat(&output, batch_size, tgt_vocab_size * TEMP_OUTPUT_SEQ_LENGTH);

  rand_mat(input, batch_size, input_dim * seq_length);

  seq2seq_inf(input, output, input_dim, seq_length, hidden_size, batch_size, tgt_vocab_size);
  return 0;
}