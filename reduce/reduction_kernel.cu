
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T>
__global__ void reduce0(T* g_idata, T* g_odata, unsigned int n) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ T sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;
  cg::sync(cta);

  // do reduction in shared mem
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <typename T>
__global__ void reduce1(T* g_idata, T* g_odata, unsigned int n) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ T sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;
  cg::sync(cta);

  // do reduction in shared mem
  for (int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T>
__global__ void reduce2(T* g_idata, T* g_odata, unsigned int n) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ T sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;
  cg::sync(cta);

  // do reduction in shared mem
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T>
__global__ void reduce3(T* g_idata, T* g_odata, unsigned int n) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ T sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;
  sdata[tid] +=  (i + blockDim.x < n) ? g_idata[i + blockDim.x] : 0;
  cg::sync(cta);

  // do reduction in shared mem
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T>
__device__ void warpReduce(volatile T* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

template <typename T>
__global__ void reduce4(T* g_idata, T* g_odata, unsigned int n) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ T sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;
  sdata[tid] +=  (i + blockDim.x < n) ? g_idata[i + blockDim.x] : 0;
  cg::sync(cta);

  // do reduction in shared mem
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  if (tid < 32)
    warpReduce(sdata, tid);

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <typename T>
void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
            T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  int smemSize = threads * sizeof(T);

  switch (whichKernel) {
    case 0:
      reduce0<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    case 1:
      reduce1<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    case 2:
      reduce2<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    case 3:
      reduce3<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    case 4:
      reduce4<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  }
}

template void reduce<float>(int size, int threads, int blocks, int whichKernel, float *d_idata, float *d_odata);