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
from tvm.topi.util import get_const_tuple
import tvm.topi.testing

import logging
import sys

import numpy as np
import tvm
from tvm import topi
from tvm import te
from tvm import autotvm

"""
Writing tunable template and Using auto-tuner
=============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_

This is an introduction tutorial to the auto-tuning module in TVM.

There are two steps in auto-tuning.
The first step is defining a search space.
The second step is running a search algorithm to explore through this space.
In this tutorial, you can learn how to perform these two steps in TVM.
The whole workflow is illustrated by a matrix multiplication example.
"""

######################################################################
# Install dependencies
# --------------------
# To use autotvm package in TVM, we need to install some extra dependencies.
# This step (installing xgboost) can be skipped as it doesn't need XGBoost
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost
#
# To make TVM run faster in tuning, it is recommended to use cython
# as FFI of TVM. In the root directory of TVM, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.


# the module is called `autotvm`


#########################################################################
# Finally we apply history best from the cache file and check its correctness.
# We can call the function :code:`matmul` directly under the
# :any:`autotvm.apply_history_best` context. When we call this function,
# it will query the dispatch context with its argument and get the best config
# with the same argument.

# apply history best from log file
def main(argv):

    args_to_str = f'({argv[1]},{argv[2]},{argv[3]})'
    print(f'm={argv[1]}, n={argv[2]}, k={argv[3]}')

    M = int(argv[1])
    N = int(argv[2])
    K = int(argv[3])
    A = te.placeholder((M, K), name='A',
                       dtype='float32')  # first tensor
    B = te.placeholder((N, K), name='B',
                       dtype='float32')  # second tensor
    task = autotvm.task.create(
        "dense_large_batch.cuda", args=(A, B), target='cuda')

    dispatch_context = autotvm.apply_history_best(f'matmul_{args_to_str}.log')
    best_config = dispatch_context.query(task.target, task.workload)
    print(best_config)

    with autotvm.apply_history_best(f'matmul_{args_to_str}.log'):
        with tvm.target.create("cuda"):
            C = topi.cuda.dense_large_batch(A, B)
            s = topi.cuda.schedule_dense_large_batch(C)
            func = tvm.build(s, [A, B, C])

    # check correctness
    a_np = np.random.uniform(size=(M, K)).astype('float32')
    b_np = np.random.uniform(size=(N, K)).astype('float32')
    c_np = np.dot(a_np, b_np.T)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    b_tvm = tvm.nd.array(b_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
    func(a_tvm, b_tvm, c_tvm)
    func.imported_modules[0].save(f"matmul_{args_to_str}.cu")

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=500)
    print('Time cost of this operator: %f' %
          evaluator(a_tvm, b_tvm, c_tvm).mean)


if __name__ == "__main__":
    main(sys.argv)
