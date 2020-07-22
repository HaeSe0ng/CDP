#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <time.h>

#include "util.h"
#include "seq2seq.h"

#define M 512
#define N 512

using namespace std;
//임베딩 구현 안함
int main()
{
  srand(0);
  int input_dim = 256;
  int seq_length = 10;
  int batch_size = 1;
  int hidden_size = 512;
  float *input, *output;
  alloc_mat(&input, batch_size, input_dim * seq_length);
  alloc_mat(&output, batch_size, input_dim * seq_length);

  rand_mat(input, batch_size, input_dim * seq_length);

  seq2seq_inf(input, output, input_dim, seq_length, hidden_size, batch_size);
  return 0;
}