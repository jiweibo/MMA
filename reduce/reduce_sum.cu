#include "common/common.h"

constexpr int MAX_BLOCK_SIZE = 1024;

__global__ void reduce0(int* g_idata, int* g_odata) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}


__global__ void reduce1(int* g_idata, int* g_odata) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    int index = tid * s * 2;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

__global__ void reduce2(int* g_idata, int* g_odata) {
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

int gpu_reduce_sum(int* d_in, int len, int warmup, int repeats) {
  int total_sum = 0;

  int block_sz = MAX_BLOCK_SIZE;
  int max_elems_per_block = block_sz;

  int grid_sz = 0;
  if (len <= max_elems_per_block) {
    grid_sz = 1;
  } else {
    grid_sz = (len + max_elems_per_block - 1) / max_elems_per_block;
  }

  int* d_block_sums;
  cudaErrCheck(cudaMalloc(&d_block_sums, sizeof(int) * grid_sz));
  cudaErrCheck(cudaMemset(d_block_sums, 0, sizeof(int) * grid_sz));

  for (int i = 0; i < warmup; ++i) {
    reduce0<<<grid_sz, block_sz, sizeof(int) * max_elems_per_block>>>(d_in, d_block_sums);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start_reduce;
  cudaEvent_t stop_reduce;
  cudaErrCheck(cudaEventCreate(&start_reduce));
  cudaErrCheck(cudaEventCreate(&stop_reduce));

  cudaErrCheck(cudaEventRecord(start_reduce));
  for (int i = 0; i < repeats; ++i) {
    reduce1<<<grid_sz, block_sz, sizeof(int) * max_elems_per_block>>>(d_in, d_block_sums);
  }
  cudaErrCheck(cudaEventRecord(stop_reduce));


  // cudaErrCheck(cudaMemcpy(&total_sum, d_total_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost));



  float time;
  cudaErrCheck(cudaEventSynchronize(stop_reduce));
  cudaErrCheck(cudaEventElapsedTime(&time, start_reduce, stop_reduce));
  std::cout << "reduce0 took " << time / repeats << " ms, " << len / (time / repeats) / 1e6 << "GFLOPS" << std::endl;

  return 0;
}