#pragma once
#include <iostream>

using namespace std;

void timer_start(int i);

double timer_stop(int i);

template <typename T> void check_mat_mul(T *A, T *B, T *C, int M, int N, int K);

template <typename T> void print_mat(T *m, int R, int C);

template <typename T> void alloc_mat(T **m, int R, int C);
template <typename T> void rand_mat(T *m, int R, int C);
void set_dis_rng(int min, int max);
template <typename T> void alloc_rand_mat(T **m, int R, int C);
template <typename T> void memset_mat(T *m, T val, int R, int C);
