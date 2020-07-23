#pragma once
#define TEMP_OUTPUT_SEQ_LENGTH 15
int seq2seq_inf(float *input, float *output, int input_dim, int seq_length, int hidden_size, int batch_size, int tgt_vocab_size);