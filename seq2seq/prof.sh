#! /bin/bash

for b in 1 4 16 64
do
  for seq_len in 16
  do
    for emb_dim in 512 # 1024
    do
      for vocab_size in 16384 # 32768
      do
        make run ARGS="$b $seq_len $emb_dim $vocab_size" >> prof_pytorch.log \
        && make run_nocdp_rdc ARGS="$b $seq_len $emb_dim $vocab_size" >> prof_pytorch.log \
        && make run_nocdp ARGS="$b $seq_len $emb_dim $vocab_size" >> prof_pytorch.log \
        && python3 seq2seq_pytorch.py $b $seq_len $emb_dim $vocab_size >> prof_pytorch.log
        #make run ARGS="$b $seq_len $emb_dim $vocab_size" >> prof.log \
        #&& make run_nocdp ARGS="$b $seq_len $emb_dim $vocab_size" >> prof.log \
        #&& make run_nocdp_rdc ARGS="$b $seq_len $emb_dim $vocab_size" >> prof.log
      done
    done
  done
done