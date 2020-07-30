#! /bin/bash

for b in 4 16 64
do
  for vocab_size in 16384 # 32768
  do
    for emb_dim in 512 # 1024
    do
      let mul_4_e=4*$emb_dim
      python3 tune_matmul_cuda.py 1 $b $mul_4_e $emb_dim && cp CUDA_Launch_LOG.log CUDA_Launch_LOG_1,$b,$mul_4_e,$emb_dim.log && python3 tune_matmul_cuda.py 1 $b $vocab_size $emb_dim && cp CUDA_Launch_LOG.log CUDA_Launch_LOG_1,$b,$vocab_size,$emb_dim.log
    done
  done
done