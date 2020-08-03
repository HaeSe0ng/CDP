#include "torch_kernels.cuh"

__global__ void indexSelectSmallIndex(float *dst, float *src, int64_t *indices,
                                      int64_t innerSize, int64_t numIndices) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (int64_t dstIndex = 0; dstIndex < numIndices; ++dstIndex) {
    int64_t srcIndex = indices[dstIndex];
    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize; linearIndex += gridDim.x * blockDim.x) {
      dst[dstIndex * innerSize + linearIndex] =
          src[srcIndex * innerSize + linearIndex];
    }
  }
}
__global__ void indexSelectLargeIndex(float *dst, float *src, int64_t *indices,
                                      int64_t totalSize, int64_t innerSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize; linearIndex += gridDim.x * blockDim.x) {
    int64_t dstIndex, elementInSlice;
    dstIndex = linearIndex / innerSize;
    elementInSlice = linearIndex % innerSize;

    int64_t srcIndex = indices[dstIndex];

    int64_t dstOffset = elementInSlice;
    dstOffset += dstIndex * innerSize;

    int64_t srcOffset = elementInSlice;
    srcOffset += srcIndex * innerSize;

    dst[dstOffset] = src[srcOffset];
  }
}
template <typename T> __device__ __forceinline__ T sigmoid(T in) {
  T one = static_cast<T>(1.0);
  return one / (one + exp(-in));
}
// bias1: input_bias, bias2: hidden_bias, cx: last cell state, hsz: hidden_size
__global__ void lstm_cell_kernel(float *input, float *hidden, float *bias1,
                                 float *bias2, float *_cx, float *_hy,
                                 float *_cy, int hsz, int totalElements) {
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements; linearIndex += gridDim.x * blockDim.x) {
    int offset = (linearIndex / hsz) * 4 * hsz + linearIndex % hsz;

    float iig = input[offset + 0 * hsz];
    float ifg = input[offset + 1 * hsz];
    float icg = input[offset + 2 * hsz];
    float iog = input[offset + 3 * hsz];

    float hig = hidden[offset + 0 * hsz];
    float hfg = hidden[offset + 1 * hsz];
    float hcg = hidden[offset + 2 * hsz];
    float hog = hidden[offset + 3 * hsz];

    float cx = _cx[linearIndex];

    float *hy = &_hy[linearIndex];
    float *cy = &_cy[linearIndex];

    float b1i, b1f, b1c, b1o;
    float b2i, b2f, b2c, b2o;

    b1i = bias1[linearIndex % hsz + 0 * hsz];
    b1f = bias1[linearIndex % hsz + 1 * hsz];
    b1c = bias1[linearIndex % hsz + 2 * hsz];
    b1o = bias1[linearIndex % hsz + 3 * hsz];

    b2i = bias2[linearIndex % hsz + 0 * hsz];
    b2f = bias2[linearIndex % hsz + 1 * hsz];
    b2c = bias2[linearIndex % hsz + 2 * hsz];
    b2o = bias2[linearIndex % hsz + 3 * hsz];

    float ig, fg, cg, og;
    float f_hy, f_cy;

    ig = sigmoid(iig + hig + b1i + b2i);
    fg = sigmoid(ifg + hfg + b1f + b2f);
    cg = tanh(icg + hcg + b1c + b2c);
    og = sigmoid(iog + hog + b1o + b2o);

    f_cy = (fg * cx) + (ig * cg);
    f_hy = og * tanh(f_cy);

    *hy = f_hy;
    *cy = f_cy;
  }
}