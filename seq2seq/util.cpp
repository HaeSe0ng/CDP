
#include <iomanip>
#include <iostream>
#include <random>

#include "util.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

using namespace std;
mt19937 gen(123);
uniform_real_distribution<float> dis_float(-0.5, 0.5);
uniform_int_distribution<int> dis_int(0, 16383);
uniform_int_distribution<int64_t> dis_int64_t(0, 16383);

static double start_time[8];

static double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) { start_time[i] = get_time(); }

double timer_stop(int i) { return get_time() - start_time[i]; }

template <typename T>
void check_mat_mul(T *A, T *B, T *C, int M, int N, int K)
{
  printf("Validating...\n");

  T *C_ans;
  alloc_mat(&C_ans, M, N);
  memset_mat<T>(C_ans, 0, M, N);
  for (int i = 0; i < M; ++i)
  {
    for (int k = 0; k < K; ++k)
    {
      for (int j = 0; j < N; ++j)
      {
        C_ans[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      T c = C[i * N + j];
      T c_ans = C_ans[i * N + j];
      if (fabsf((float)(c - c_ans)) > eps &&
          (c_ans == 0 || fabsf((float)(c - c_ans) / c_ans) > eps))
      {
        ++cnt;
        if (cnt <= thr)
          printf("C[%d][%d]", i, j);
        cout << " : correct_value = " << c_ans << ", your_value = " << c
             << endl;
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  if (is_valid)
  {
    printf("Result: VALID\n");
  }
  else
  {
    printf("Result: INVALID\n");
  }
}
template <typename T>
void print_mat(T *m, int R, int C)
{
  cout.setf(ios::fixed);
  for (int i = 0; i < R; ++i)
  {
    for (int j = 0; j < C; ++j)
    {
      cout << setprecision(3) << m[i * C + j] << " ";
    }
    printf("\n");
  }
}

template <typename T>
void alloc_mat(T **m, int R, int C)
{
  *m = (T *)malloc(sizeof(T) * R * C);
  if (*m == NULL)
  {
    printf("Failed to allocate memory for matrix.\n");
    exit(0);
  }
}
template <typename T>
void rand_mat(T *m, int R, int C)
{
  for (int i = 0; i < R; i++)
  {
    for (int j = 0; j < C; j++)
    {
      m[i * C + j] = (T)dis_float(gen);
    }
  }
}
template <>
void rand_mat(int *m, int R, int C)
{
  for (int i = 0; i < R; i++)
  {
    for (int j = 0; j < C; j++)
    {
      m[i * C + j] = dis_int(gen);
    }
  }
}
template <>
void rand_mat(int64_t *m, int R, int C)
{
  for (int i = 0; i < R; i++)
  {
    for (int j = 0; j < C; j++)
    {
      m[i * C + j] = dis_int64_t(gen);
    }
  }
}
void set_dis_rng(int64_t min, int64_t max)
{
  dis_int64_t(gen, decltype(dis_int64_t)::param_type(min, max));
}
template <typename T>
void alloc_rand_mat(T **m, int R, int C)
{
  alloc_mat(m, R, C);
  rand_mat(*m, R, C);
}
template <typename T>
void memset_mat(T *m, T val, int R, int C)
{
  memset(m, val, sizeof(T) * R * C);
}

template void check_mat_mul<int>(int *A, int *B, int *C, int M, int N, int K);
template void check_mat_mul<int64_t>(int64_t *A, int64_t *B, int64_t *C, int M,
                                     int N, int K);
template void check_mat_mul<float>(float *A, float *B, float *C, int M, int N,
                                   int K);

template void print_mat<int>(int *m, int R, int C);
template void print_mat<int64_t>(int64_t *m, int R, int C);
template void print_mat<float>(float *m, int R, int C);

template void alloc_mat<int>(int **m, int R, int C);
template void alloc_mat<int64_t>(int64_t **m, int R, int C);
template void alloc_mat<float>(float **m, int R, int C);

template void rand_mat<float>(float *m, int R, int C);

template void alloc_rand_mat<int>(int **m, int R, int C);
template void alloc_rand_mat<int64_t>(int64_t **m, int R, int C);
template void alloc_rand_mat<float>(float **m, int R, int C);

template void memset_mat<int>(int *m, int val, int R, int C);
template void memset_mat<int64_t>(int64_t *m, int64_t val, int R, int C);
template void memset_mat<float>(float *m, float val, int R, int C);