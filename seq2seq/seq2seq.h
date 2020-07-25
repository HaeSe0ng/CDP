#pragma once

int seq2seq_inf(int *input, int *output, int sos, int *eos,
                int emb_dim, int seq_length, int hidden_size, int batch_size,
                int src_vocab_size, int tgt_vocab_size, int max_len);