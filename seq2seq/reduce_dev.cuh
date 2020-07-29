// Copied from aten/src/ATen/native/cuda/Reduce.cuh

#pragma once
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iosfwd>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <tuple>
#include <type_traits>
#include <utility>

#define FLT_MAX __FLT_MAX__
#define DBL_MAX __DBL_MAX__

#define MAX(A, B) A >= B ? A : B
#define MIN(A, B) A <= B ? A : B

constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;

// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block
// size. 256 is a good number for this fallback and should give good occupancy
// and versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
#define C10_MAX_THREADS_PER_BLOCK(val)           \
  (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                         : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)        \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block)-1) /         \
           (threads_per_block))))

#define C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) \
  __launch_bounds__(                                                  \
      (C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))),           \
      (C10_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))
#define C10_WARP_SIZE 32

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector
{
  scalar_t val[vec_size];
};
namespace at
{
  template <typename T>
  struct numeric_limits
  {
  };
  template <>
  struct numeric_limits<float>
  {
    static inline __device__ float lowest() { return -FLT_MAX; }
    static inline __device__ float max() { return FLT_MAX; }
    static inline __device__ float lower_bound()
    {
      return -static_cast<float>(INFINITY);
    }
    static inline __device__ float upper_bound()
    {
      return static_cast<float>(INFINITY);
    }
  };
} // namespace at
struct nullopt_t
{
  struct init
  {
  };
  constexpr explicit nullopt_t(init) {}
};
constexpr nullopt_t nullopt{nullopt_t::init()};
template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta,
                                            int width = warpSize,
                                            unsigned int mask = 0xffffffff)
{
  return __shfl_down_sync(0xFFFFFFFF, value, delta, width);
}
namespace detail
{

  template <typename T, int size>
  struct alignas(16) Array
  {
    T data[size];

    __device__ T operator[](int i) const { return data[i]; }
    __device__ T &operator[](int i) { return data[i]; }
    Array() = default;
    Array(const Array &) = default;
    Array &operator=(const Array &) = default;

    // Fill the array with x.
    __device__ Array(T x)
    {
      for (int i = 0; i < size; i++)
      {
        data[i] = x;
      }
    }
  };

  template <typename T1, typename T2>
  using pair = thrust::pair<T1, T2>;

  template <typename scalar_t>
  struct LessOrNan
  {
    __device__ bool operator()(scalar_t a, scalar_t b) const
    {
      return isnan(a) || a < b;
    }
  };

  template <typename scalar_t>
  struct GreaterOrNan
  {
    __device__ bool operator()(scalar_t a, scalar_t b) const
    {
      return isnan(a) || a > b;
    }
  };

  template <typename comp_t>
  struct MinMaxReductionOps
  {
    using scalar_t = float;
    using index_t = int64_t;
    using arg_t = detail::pair<scalar_t, index_t>;

    static __device__ arg_t project(arg_t arg) { return arg; }

    static __device__ arg_t reduce(arg_t arg, scalar_t val, int64_t idx)
    {
      return comp_t{}(arg.first, val) ? arg : arg_t(val, idx);
    }

    static __device__ arg_t combine(arg_t a, arg_t b)
    {
      return comp_t{}(a.first, b.first) ? a : b;
    }

    static __device__ arg_t translate_idx(arg_t a, int64_t base_idx)
    {
      return {a.first, a.second + base_idx};
    }
    static __device__ arg_t warp_shfl_down(arg_t arg, int offset)
    {
      return arg_t(WARP_SHFL_DOWN(arg.first, offset),
                   WARP_SHFL_DOWN(arg.second, offset));
    }
  };

  template <typename comp_t>
  struct ArgReductionOps : public MinMaxReductionOps<comp_t>
  {
    using typename MinMaxReductionOps<comp_t>::scalar_t;
    using typename MinMaxReductionOps<comp_t>::index_t;
    using typename MinMaxReductionOps<comp_t>::arg_t;

    static __device__ index_t project(arg_t arg) { return arg.second; }
  };

} // namespace detail

template <typename scalar_t>
struct ArgMaxOps
    : public detail::ArgReductionOps<detail::GreaterOrNan<scalar_t>>
{
};

static inline __device__ int64_t div_up(int64_t a, int64_t b) { return (a + b - 1) / b; }

// returns floor(log2(n))
static inline __device__ int last_pow2(int n)
{
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return MAX(1, n - (n >> 1));
}

// returns reduced fraction numerator & denominator
static __device__ void reduce_fraction(size_t &numerator,
                                       size_t &denominator)
{
  // get GCD of num and denom using Euclid's algorithm.
  // Can replace this with std::gcd if we ever support c++17.
  size_t a = denominator;
  size_t b = numerator;
  while (b != 0)
  {
    a %= b;
    // swap(a,b)
    size_t tmp = a;
    a = b;
    b = tmp;
  }

  // a is now the GCD
  numerator /= a;
  denominator /= a;
}

struct ReduceConfig
{
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  static constexpr int MAX_NUM_THREADS = 512;
  static constexpr int input_vec_size = 4;

  __device__ ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
      : element_size_bytes(element_size_bytes), num_inputs(num_inputs),
        num_outputs(num_outputs) {}

  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  __device__ void set_block_dimension(int64_t dim0, int64_t dim1)
  {
    int dim0_pow2 = dim0 < MAX_NUM_THREADS ? static_cast<int>(last_pow2(dim0))
                                           : MAX_NUM_THREADS;
    int dim1_pow2 = dim1 < MAX_NUM_THREADS ? static_cast<int>(last_pow2(dim1))
                                           : MAX_NUM_THREADS;
    block_width = MIN(dim0_pow2, C10_WARP_SIZE);
    block_height = MIN(dim1_pow2, int(MAX_NUM_THREADS / block_width));
    block_width = MIN(dim0_pow2, int(MAX_NUM_THREADS / block_height));
    num_threads = block_width * block_height;
  }

  __device__ int split_input(int parallelism)
  {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  __device__ int split_output(int parallelism)
  {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  __device__ dim3 block() const { return dim3(block_width, block_height); }

  __device__ dim3 grid() const
  {
    return dim3(div_up(num_outputs, step_output),
                ctas_per_output);
  }

  __device__ bool should_block_x_reduce() const
  {
    return input_mult[BLOCK_X] != 0;
  }

  __device__ bool should_block_y_reduce() const
  {
    return input_mult[BLOCK_Y] != 0;
  }

  __device__ bool should_global_reduce() const
  {
    return input_mult[CTA] != 0;
  }

  __device__ bool should_store(int output_idx) const
  {
    return output_idx < num_outputs &&
           (!should_block_x_reduce() || threadIdx.x == 0) &&
           (!should_block_y_reduce() || threadIdx.y == 0);
  }

  __device__ bool should_reduce_tail() const
  {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
           (!should_global_reduce() || blockIdx.y == 0);
  }

  __device__ int input_idx() const
  {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] + warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  __device__ int output_idx() const
  {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] + warp * output_mult[BLOCK_Y] +
            cta1 * step_output);
  }

  __device__ int shared_memory_offset(int offset) const
  {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  __device__ int staging_memory_offset(int cta2) const
  {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce())
    {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  __device__ int shared_memory_size() const
  {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() || block_width <= C10_WARP_SIZE))
    {
      return 0;
    }
    return element_size_bytes * num_threads;
  }

  __device__ int64_t global_memory_size() const
  {
    if (!should_global_reduce())
    {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce())
    {
      size *= block().x;
    }
    return size;
  }

  __device__ int semaphore_size() const
  {
    if (!should_global_reduce())
    {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  __device__ int values_per_thread() const { return div_up(num_inputs, step_input); }
};

std::ostream &operator<<(std::ostream &out, const ReduceConfig &config);

template <int nt, typename R>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void reduce_kernel(R reduction)
{
  reduction.run();
}

constexpr int MAX_DIMS = 25;
// Result of div/mod operation stored together.
template <typename Value>
struct DivMod
{
  Value div, mod;

  __device__ DivMod(Value div, Value mod) : div(div), mod(mod) {}
};

// Base case: we only have an implementation for uint32_t for now.  For
// everything else, we use plain division.
template <typename Value>
struct IntDivider
{
  __device__ IntDivider() {} // Dummy constructor for arrays.
  __device__ IntDivider(Value d) : divisor(d) {}

  __device__ inline Value div(Value n) const { return n / divisor; }
  __device__ inline Value mod(Value n) const { return n % divisor; }
  __device__ inline DivMod<Value> divmod(Value n) const
  {
    return DivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};
template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator
{
  // The offset for each argument (in bytes). Wrapper around fixed-size array.
  using offset_type = detail::Array<index_t, NARGS>;

  __device__ OffsetCalculator(int dims, const int64_t *sizes,
                              const int64_t *const *strides)
      : dims(dims)
  {
    //if (dims > MAX_DIMS)
    //fprintf(stderr, "tensor has too many (>%d) dims", MAX_DIMS);
    for (int i = 0; i < MAX_DIMS; ++i)
    {
      if (i < dims)
      {
        sizes_[i] = IntDivider<index_t>(sizes[i]);
      }
      else
      {
        sizes_[i] = IntDivider<index_t>(1);
      }
      for (int arg = 0; arg < NARGS; arg++)
      {
        strides_[i][arg] = i < dims ? strides[arg][i] : 0;
      }
    }
  }

  __device__ offset_type get(index_t linear_idx) const
  {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++)
    {
      offsets[arg] = 0;
    }

#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim)
    {
      if (dim == dims)
      {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

#pragma unroll
      for (int arg = 0; arg < NARGS; arg++)
      {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }
    }
    return offsets;
  }

  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS][NARGS];
};

template <int vt, typename index_t, typename func_t>
__device__ void strided_iterate(func_t f, index_t begin, index_t end,
                                index_t stride)
{
  if (begin + (vt - 1) * stride < end)
  {
#pragma unroll
    for (index_t i = 0; i < vt; i++)
    {
      f(i, begin + i * stride);
    }
  }
  else
  {
#pragma unroll
    for (index_t i = 0; i < vt; i++)
    {
      index_t idx = begin + i * stride;
      if (idx < end)
      {
        f(i, idx);
      }
    }
  }
}

// template <typename out_scalar_t, typename func_t>
// struct func_wrapper_t {
//   using arg_t = typename binary_function_traits<func_t>::arg1_t;
//   using scalar_t = typename binary_function_traits<func_t>::arg2_t;

//   func_t combine;
//   static inline __device__ out_scalar_t project(arg_t arg) {
//     return (out_scalar_t) arg;
//   }
//   static inline __device__ arg_t warp_shfl_down(arg_t arg, int offset) {
//     return WARP_SHFL_DOWN(arg, offset);
//   }

//   func_wrapper_t(const func_t& op) : combine(op) {
//   }

//   // wrap a normal reduction that ignores the index
//   __device__ arg_t reduce(arg_t acc, scalar_t val, int64_t idx) const {
//     return combine(acc, val);
//   }
// };

// template <typename scalar_t, typename func_t>
// func_wrapper_t<scalar_t, func_t> func_wrapper(const func_t& op) {
//   return func_wrapper_t<scalar_t, func_t> { op };
// }

template <typename scalar_t, typename ops_t, typename index_t, typename out_scalar_t = scalar_t, int vt0 = 4>
struct ReduceOp
{
  using arg_t = detail::pair<float, int64_t>;

  using InputCalculator = OffsetCalculator<1, index_t>;
  using OutputCalculator = OffsetCalculator<2, index_t>;

  static constexpr bool can_accumulate_in_output =
      std::is_convertible<arg_t, out_scalar_t>::value && std::is_convertible<out_scalar_t, arg_t>::value;

  static constexpr float acc_buffer_multiplier = (float)sizeof(arg_t) / sizeof(out_scalar_t);

  ops_t ops;
  arg_t ident;
  ReduceConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void *src;
  const char *dst[2]; //it accepts at most two destinations
  // acc_buf used for accumulation among sub Tensor Iterator when accumulation on
  // output is not permissible
  void *acc_buf;
  // cta_buf used for accumulation between blocks during global reduction
  void *cta_buf;
  int *semaphores;
  bool accumulate;
  bool final_output;
  int noutputs;

  __device__ ReduceOp(ops_t ops, ReduceConfig config, InputCalculator input_calc, OutputCalculator output_calc,
                      const void *src, char *dst0, void *acc_buf, void *cta_buf, int *semaphores, arg_t ident, int noutputs)
      : ops(ops), config(config), input_calc(input_calc), output_calc(output_calc), src(src), acc_buf(acc_buf), cta_buf(cta_buf), semaphores(semaphores), ident(ident), noutputs(noutputs)
  {
    dst[0] = dst0;
  }

  __device__ void run() const
  {
    extern __shared__ char shared_memory[];
    index_t output_idx = config.output_idx();
    index_t input_idx = config.input_idx();
    auto base_offsets = output_calc.get(output_idx);

    arg_t value = ident;
    if (output_idx < config.num_outputs && input_idx < config.num_inputs)
    {
      auto input_slice = (const char *)src + base_offsets[1];
      value = thread_reduce((const scalar_t *)input_slice);
    }

    if (config.should_block_y_reduce())
    {
      value = block_y_reduce(value, shared_memory);
    }
    if (config.should_block_x_reduce())
    {
      value = block_x_reduce(value, shared_memory);
    }

    auto out = (out_scalar_t *)((char *)dst[0] + base_offsets[0]);
    arg_t *acc = nullptr;
    if (acc_buf != nullptr)
    {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(numerator, denominator);
      acc = (arg_t *)((char *)acc_buf + (base_offsets[0] * numerator / denominator));
    }

    if (config.should_global_reduce())
    {
      value = global_reduce(value, acc, shared_memory);
    }
    else if (config.should_store(output_idx))
    {
      if (acc == nullptr)
      {
        if (accumulate)
        {
          value = accumulate_in_output<can_accumulate_in_output>(out, value);
        }
        if (final_output)
        {
          set_results_to_output(value, base_offsets[0]);
        }
        else
        {
          *out = get_accumulated_output<can_accumulate_in_output>(out, value);
        }
      }
      else
      {
        if (accumulate)
        {
          value = ops.combine(*acc, value);
        }
        if (final_output)
        {
          set_results_to_output(value, base_offsets[0]);
        }
        else
        {
          *acc = value;
        }
      }
    }
  }

  __device__ arg_t thread_reduce(const scalar_t *data) const
  {
    index_t idx = config.input_idx();
    // Multiple accumulators to remove dependency between unrolled loops.
    arg_t value_list[vt0];
#pragma unroll
    for (int i = 0; i < vt0; i++)
    {
      value_list[i] = ident;
    }
    index_t end = config.num_inputs;
    index_t stride = config.step_input;
    index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);

    // Reducing layers of function calls so compiler could do proper loop unroll
    // that exposes instruction level parallelism.
    while (idx < config.num_inputs)
    {
      // load input
      detail::Array<scalar_t, vt0> values;
      if (input_calc.dims == 1)
      {
        strided_iterate<vt0>([&](index_t i, index_t idx) {
          values[i] = data[idx * element_stride];
        },
                             idx, end, stride);
      }
      else
      {
        strided_iterate<vt0>([&](index_t i, index_t idx) {
          values[i] = data[input_calc.get(idx)[0] / sizeof(scalar_t)];
        },
                             idx, end, stride);
      }
      // compute
      strided_iterate<vt0, index_t>([&](index_t i, index_t idx) {
        value_list[i] = ops.reduce(value_list[i], values[i], idx);
      },
                                    idx, config.num_inputs, config.step_input);
      // step offset
      idx += config.step_input * vt0;
    }
#pragma unroll
    for (int i = 1; i < vt0; i++)
    {
      value_list[0] = ops.combine(value_list[0], value_list[i]);
    }
    return value_list[0];
  }

  __device__ arg_t block_x_reduce(arg_t value, char *shared_memory) const
  {
    int dim_x = blockDim.x;
    arg_t *shared = (arg_t *)shared_memory;
    if (dim_x > warpSize)
    {
      int address_base = threadIdx.x + threadIdx.y * blockDim.x;
      shared[address_base] = value;
      for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1)
      {
        __syncthreads();
        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x)
        {
          arg_t other = shared[address_base + offset];
          value = ops.combine(value, other);
          shared[address_base] = value;
        }
      }
      dim_x = warpSize;
    }

    __syncthreads();

    for (int offset = 1; offset < dim_x; offset <<= 1)
    {
      arg_t other = ops.warp_shfl_down(value, offset);
      value = ops.combine(value, other);
    }
    return value;
  }

  __device__ arg_t block_y_reduce(arg_t value, char *shared_memory) const
  {
    arg_t *shared = (arg_t *)shared_memory;
    shared[config.shared_memory_offset(0)] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1)
    {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y)
      {
        arg_t other = shared[config.shared_memory_offset(offset)];
        value = ops.combine(value, other);
        shared[config.shared_memory_offset(0)] = value;
      }
    }
    return value;
  }

  __device__ bool mark_block_finished() const
  {
    __shared__ bool is_last_block_done_shared;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
    }

    __syncthreads();

    return is_last_block_done_shared;
  }

  template <bool can_acc>
  __device__ arg_t accumulate_in_output(
      out_scalar_t *out, arg_t value,
      typename std::enable_if<can_acc>::type * = nullptr) const
  {
    return ops.combine(*out, value);
  }

  template <bool can_acc>
  __device__ out_scalar_t get_accumulated_output(
      out_scalar_t *out, arg_t value,
      typename std::enable_if<can_acc>::type * = nullptr) const
  {
    assert(!final_output);
    return (out_scalar_t)value;
  }

  // This function should never be called --
  // it's the version of `accumulate_in_output`
  // when accumulation in the output is not possible.
  template <bool can_acc>
  __device__ arg_t accumulate_in_output(
      out_scalar_t *, arg_t,
      typename std::enable_if<!can_acc>::type * = nullptr) const
  {
    assert(false); // can't use AT_ASSERT in Cuda.
    return arg_t{};
  }

  // This function should never be called --
  // it's the version of `get_accumulated_output`
  // when accumulation in the output is not possible.
  template <bool can_acc>
  __device__ out_scalar_t get_accumulated_output(
      out_scalar_t *out, arg_t value,
      typename std::enable_if<!can_acc>::type * = nullptr) const
  {
    assert(false);
    return *out;
  }

  template <class T>
  __device__ void set_results(const T x, const index_t base_offset) const
  {
    assert(noutputs == 1);
    auto res = (out_scalar_t *)((char *)dst[0] + base_offset);
    *res = x;
  }

  //   //Currently implemented for max of two outputs
  //   template<class T>
  //   __device__ void set_results(const thrust::tuple<T, T> x, const index_t base_offset) const {
  //     if (noutputs >= 1) {
  //       auto res0 = (out_scalar_t*)((char*)dst[0] + base_offset);
  //       *res0 = thrust::get<0>(x);
  //     }
  //     if (noutputs >= 2) {
  //       auto res1 = (out_scalar_t *) ((char *) dst[1] + base_offset);
  //       *res1 = thrust::get<1>(x);
  //     }
  //   }

  __device__ void set_results_to_output(arg_t value, index_t base_offset) const
  {
    assert(final_output);
    set_results(ops.project(value), base_offset);
  }

  __device__ arg_t global_reduce(arg_t value, arg_t *acc, char *shared_memory) const
  {
    arg_t *reduce_buffer = (arg_t *)cta_buf;
    index_t output_idx = config.output_idx();
    auto base_offsets = output_calc.get(output_idx);
    auto out = (out_scalar_t *)((char *)dst[0] + base_offsets[0]);

    bool should_store = config.should_store(config.output_idx());
    if (should_store)
    {
      index_t offset = config.staging_memory_offset(blockIdx.y);
      reduce_buffer[offset] = value;
    }

    __threadfence(); // make sure writes are globally visible
    __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done
    bool is_last_block_done = mark_block_finished();

    if (is_last_block_done)
    {
      value = ident;
      if (config.should_block_x_reduce())
      {
        index_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
        index_t step = blockDim.x * blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step)
        {
          index_t idx = config.staging_memory_offset(input_offset);
          arg_t next = reduce_buffer[idx];
          value = ops.combine(value, next);
        }
      }
      else
      {
        index_t input_offset = threadIdx.y;
        index_t step = blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step)
        {
          index_t idx = config.staging_memory_offset(input_offset);
          arg_t next = reduce_buffer[idx];
          value = ops.combine(value, next);
        }
      }
      value = block_y_reduce(value, shared_memory);
      if (config.should_block_x_reduce())
      {
        value = block_x_reduce(value, shared_memory);
      }
      if (should_store)
      {
        if (acc == nullptr)
        {
          if (accumulate)
          {
            value = accumulate_in_output<can_accumulate_in_output>(out, value);
          }
          if (final_output)
          {
            set_results_to_output(value, base_offsets[0]);
          }
          else
          {
            *out = get_accumulated_output<can_accumulate_in_output>(out, value);
          }
        }
        else
        {
          if (accumulate)
          {
            value = ops.combine(*acc, value);
          }
          if (final_output)
          {
            set_results_to_output(value, base_offsets[0]);
          }
          else
          {
            *acc = value;
          }
        }
      }
    }

    return value;
  }
};

template <int nt, typename R>
static __device__ void launch_reduce_kernel(const ReduceConfig &config, const R &reduction)
{
  dim3 block = config.block();
  dim3 grid = config.grid();

  int shared_memory = config.shared_memory_size();
  reduce_kernel<nt, R><<<grid, block, shared_memory>>>(reduction);
  //   AT_CUDA_CHECK(cudaGetLastError());
}

// struct AccumulationBuffer {
//   AccumulationBuffer() {}

//   AccumulationBuffer(size_t acc_t_size, size_t out_t_size, char* out_ptr,
//   int64_t size) {
//     out_ptr_ = (char*)out_ptr;
//     if (out_t_size >= acc_t_size) {
//       // reusing output buffer for accumulation.
//       acc_ptr_ = (char*)out_ptr;
//       numerator_ = 1;
//       denominator_ = 1;
//     } else {
//       auto& allocator = *globalContext().getTHCState()->cudaDeviceAllocator;
//       buffer_ = allocator.allocate(size);
//       acc_ptr_ = (char*)buffer_.get();
//       numerator_ = acc_t_size;
//       denominator_ = out_t_size;
//       reduce_fraction(numerator_, denominator_);
//     }
//   }

//   char* get_acc_slice(char* out_ptr) {
//     if (numerator_ == -1 || acc_ptr_ == nullptr) {
//       return nullptr;
//     }
//     return acc_ptr_ + ((out_ptr - out_ptr_) * numerator_ / denominator_);
//   }

//   char* acc_ptr_ = nullptr;
//   char* out_ptr_ = nullptr;
//   float size_factor_ = -1;
//   size_t numerator_ = -1;
//   size_t denominator_ = -1;
//   DataPtr buffer_;
// };
