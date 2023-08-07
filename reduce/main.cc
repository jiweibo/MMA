#include <cstdlib>
#include "reduce_sum.h"
#include "common/common.h"


void generate_input(int* input, int len) {
  for (int i = 0; i < len; ++i) {
    input[i] = i;
  }
}

int main(int argc, char** argv) {
  const int warmup = 0;
  const int repeats = 1;

  int N = (1 << 20) * 4;

  int* host_in = new int[N];
  generate_input(host_in, N);

  int* dev_in;
  cudaErrCheck(cudaMalloc(&dev_in, sizeof(int) * N));
  cudaErrCheck(cudaMemcpy(dev_in, host_in, sizeof(int) * N, cudaMemcpyHostToDevice));

  gpu_reduce_sum(dev_in, N, warmup, repeats);

  return 0;
}