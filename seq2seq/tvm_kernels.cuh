__global__ void
batch_matmul_1_256_1_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                void *__restrict__ compute);
__global__ void
batch_matmul_1_2048_1_256_kernel(void *__restrict__ A, void *__restrict__ B,
                                 void *__restrict__ compute);
__global__ void
batch_matmul_1_2048_1_512_kernel(void *__restrict__ A, void *__restrict__ B,
                                 void *__restrict__ compute);