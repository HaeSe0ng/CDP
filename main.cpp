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
  int input_dim = 2048;
  int seq_length = 10;
  int batch_size = 1;
  int hidden_size = 1024;
  float *input, *output;
  alloc_mat(&input, input_dim * seq_length, 1);
  alloc_mat(&output, input_dim * seq_length, 1);

  rand_mat(input, input_dim * seq_length, 1);

  seq2seq_inf(input, output, input_dim, seq_length, hidden_size, batch_size);
  return 0;
}