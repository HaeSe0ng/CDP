# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
import topi.testing

import logging
import sys

import numpy as np
import tvm
import topi
from tvm import te

# the module is called `autotvm`
from tvm import autotvm

import sys


def verify_batch_matmul(batch, M, N, K):
    x = te.placeholder((batch, M, K), name='x')
    y = te.placeholder((batch, N, K), name='y')
    dtype = x.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_batch_matmul")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, M, K)).astype(dtype)
        b_np = np.random.uniform(size=(batch, N, K)).astype(dtype)
        c_np = topi.testing.batch_matmul(a_np, b_np)
        return (a_np, b_np, c_np)
    # get the test data
    a_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            fcompute, fschedule = topi.testing.dispatch(
                device, _batch_matmul_implement)
            out = fcompute(x, y)
            s = fschedule([out])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(
            np.zeros(get_const_tuple(out.shape), dtype=dtype), ctx)
        f = tvm.build(s, [x, y, out], device, name="dense")
        f(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    check_device('cuda')


def main(argv):

    args_to_str = f'({argv[1]},{argv[2]},{argv[3]},{argv[4]})'
    print(f'bsz={argv[1]}, m={argv[2]}, n={argv[3]}, k={argv[4]}')

    _batch_matmul_implement = {
        "gpu": (topi.cuda.batch_matmul, topi.cuda.schedule_batch_matmul),
    }

    bsz = int(argv[1])
    M = int(argv[2])
    N = int(argv[3])
    K = int(argv[4])
    A = te.placeholder((bsz, M, K), name='A',
                       dtype='float32')  # first tensor
    B = te.placeholder((bsz, N, K), name='B',
                       dtype='float32')  # second tensor
    task = autotvm.task.create(
        "batch_matmul.cuda", args=(A, B), target='cuda')
    print(task.config_space)

    ################################################################
    # Then we need to define how to measure the generated code and pick a tuner.
    # Since our space is small, a random tuner is just okay.
    #
    # We only make 10 trials in this tutorial for demonstration. In practice,
    # you can do more trials according to your time budget.
    # We will log the tuning results into a log file. This file can be
    # used to get the best config later.

    # logging config (for printing tuning log to the screen)
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    # There are two steps for measuring a config: build and run.
    # By default, we use all CPU cores to compile program. Then measure them sequentially.
    # We measure 5 times and take average to reduce variance.
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=150, timeout=4)
    )

    # Begin tuning with RandomTuner, log records to file `matmul.log`
    # You can use alternatives like XGBTuner.
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=1000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(f'matmul_{args_to_str}.log')])

    #########################################################################
    # Finally we apply history best from the cache file and check its correctness.
    # We can call the function :code:`matmul` directly under the
    # :any:`autotvm.apply_history_best` context. When we call this function,
    # it will query the dispatch context with its argument and get the best config
    # with the same argument.

    # apply history best from log file
    dispatch_context = autotvm.apply_history_best(f'matmul_{args_to_str}.log')
    best_config = dispatch_context.query(task.target, task.workload)
    print(best_config)

    with autotvm.apply_history_best(f'matmul_{args_to_str}.log'):
        with tvm.target.create("cuda"):
            C = topi.cuda.batch_matmul(A, B)
            s = topi.cuda.schedule_batch_matmul(C)
            func = tvm.build(s, [A, B, C])

    # check correctness
    a_np = np.random.uniform(size=(bsz, M, K)).astype('float32')
    b_np = np.random.uniform(size=(bsz, N, K)).astype('float32')
    c_np = topi.testing.batch_matmul(a_np, b_np)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    b_tvm = tvm.nd.array(b_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
    func(a_tvm, b_tvm, c_tvm)
    func.imported_modules[0].save(f"matmul_{args_to_str}.cu")

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=400)
    print('Time cost of this operator: %f' %
          evaluator(a_tvm, b_tvm, c_tvm).mean)


if __name__ == "__main__":
    main(sys.argv)
