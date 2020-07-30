#pragma once

#include <iomanip>

int seq2seq_inf(int64_t *input, int64_t *output, int64_t sos, int64_t *eos,
                int emb_dim, int seq_length, int hidden_size, int batch_size,
                int src_vocab_size, int tgt_vocab_size, int max_len, float *res);