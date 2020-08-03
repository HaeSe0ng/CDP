#pragma once
#include <cuda_runtime.h>

#define AT_APPLY_THREADS_PER_BLOCK 512
#define THCCeilDiv(a, b) (a + b - 1) / b

__global__ void indexSelectSmallIndex(float *dst, float *src, int64_t *indices,
                                      int64_t innerSize, int64_t numIndices);
__global__ void indexSelectLargeIndex(float *dst, float *src, int64_t *indices,
                                      int64_t totalSize, int64_t innerSize);
__global__ void lstm_cell_kernel(float *input, float *hidden, float *bias1,
                                 float *bias2, float *_cx, float *_hy,
                                 float *_cy, int hsz, int totalElements);