import numpy as np
import torch
import sys

max_len = 300
batch_size = int(sys.argv[1])
seq_length = int(sys.argv[2])
emb_dim = int(sys.argv[3])
hidden_size = emb_dim
vocab_size = int(sys.argv[4])

torch.cuda.is_available()
input = torch.randint(
    vocab_size, (seq_length, batch_size)).cuda()
weight_ih = torch.randn(4*hidden_size, emb_dim).cuda()
weight_hh = torch.randn(4*hidden_size, hidden_size).cuda()
bias_ih = torch.randn(4*hidden_size).cuda()
bias_hh = torch.randn(4*hidden_size).cuda()
hx = torch.zeros(batch_size, hidden_size, dtype=torch.float).cuda()
cx = torch.zeros(batch_size, hidden_size, dtype=torch.float).cuda()
weight_ho = torch.randn(vocab_size, hidden_size).cuda()
sos_batch = torch.zeros(batch_size, dtype=torch.int64).cuda()
output_saved = torch.zeros(seq_length, batch_size).pin_memory()
output_cpu = torch.zeros(seq_length, batch_size).pin_memory()

with torch.no_grad():
    emb = torch.nn.Embedding(vocab_size, hidden_size).cuda()
    lstm = torch.nn.LSTMCell(hidden_size, hidden_size, True).cuda()
    fc = torch.nn.Linear(hidden_size, vocab_size).cuda()

    lstm.weight_ih = torch.nn.Parameter(weight_ih)
    lstm.weight_hh = torch.nn.Parameter(weight_hh)
    lstm.bias_ih = torch.nn.Parameter(bias_ih)
    lstm.bias_hh = torch.nn.Parameter(bias_hh)
    fc.weight = torch.nn.Parameter(weight_ho)

    start = torch.cuda.Event(enable_timing=True)
    encode = torch.cuda.Event(enable_timing=True)
    decode = torch.cuda.Event(enable_timing=True)
    memcpy = torch.cuda.Event(enable_timing=True)

    # encode
    for i in range(seq_length):
        emb_vec = emb(input[i])
        hx, cx = lstm(emb_vec, (hx, cx))
    enc_hx_out, enc_cx_out = hx, cx
    # decode (save output)
    output = sos_batch
    for i in range(seq_length):
        emb_vec = emb(output)
        hx, cx = lstm(emb_vec, (hx, cx))
        output_onehot = fc(hx)
        output = output_onehot.argmax(1)
        output_saved[i] = output

    elapsed_time_enc = 0
    elapsed_time_dec = 0
    elapsed_time_mem = 0
    num_itr = 200
    for itr in range(-num_itr, num_itr):
        hx = torch.zeros(batch_size, hidden_size, dtype=torch.float).cuda()
        cx = torch.zeros(batch_size, hidden_size, dtype=torch.float).cuda()
        # warmup
        if(itr < 0):
            for i in range(seq_length):
                emb_vec = emb(input[i])
                hx, cx = lstm(emb_vec, (hx, cx))

            output = sos_batch
            for i in range(seq_length):
                emb_vec = emb(output)
                hx, cx = lstm(emb_vec, (hx, cx))
                output_onehot = fc(hx)
                output = output_onehot.argmax(1)

        # measure time
        else:
            start.record()
            # encode
            for i in range(seq_length):
                emb_vec = emb(input[i])
                hx, cx = lstm(emb_vec, (hx, cx))

            encode.record()
            encode.synchronize()
            elapsed_time_enc += start.elapsed_time(encode)

            # decode
            output = sos_batch
            start.record()
            for i in range(max_len):
                emb_vec = emb(output)
                hx, cx = lstm(emb_vec, (hx, cx))
                output_onehot = fc(hx)
                output = output_onehot.argmax(1)
                decode.record()
                output_cpu[i].copy_(output, non_blocking=True)
                memcpy.record()
                memcpy.synchronize()
                elapsed_time_mem += decode.elapsed_time(memcpy)
                if torch.all(torch.eq(output_saved[seq_length-1], output_cpu[i])):
                    break

            elapsed_time_dec += start.elapsed_time(memcpy)
    elapsed_time_enc /= num_itr
    elapsed_time_dec /= num_itr
    elapsed_time_mem /= num_itr
    print(elapsed_time_enc + elapsed_time_dec)
    print(elapsed_time_enc)
    print(elapsed_time_dec)
    print(elapsed_time_mem)
    print(elapsed_time_dec - elapsed_time_mem)
