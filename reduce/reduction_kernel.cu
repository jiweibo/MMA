
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

template <typename T, unsigned int blockSize>
__device__ void warpReduce2(volatile T* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template <typename T, unsigned int blockSize>
__global__ void reduce5(T* g_idata, T* g_odata, unsigned int n) {
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ T sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    sdata[tid] += (i + blockDim.x < n) ? g_idata[i + blockDim.x] : 0;
    cg::sync(cta);

    if (blockSize == 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        cg::sync(cta);
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        cg::sync(cta);
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        cg::sync(cta);
    }
    if (blockSize >= 128) {
        if (tid < 64)   sdata[tid] += sdata[tid + 64];
        cg::sync(cta);
    }
    if (tid < 32) {
        warpReduce2<T, blockSize>(sdata, tid);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T, unsigned int blockSize>
__global__ void reduce6(T* g_idata, T* g_odata, unsigned int n) {
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ T sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    cg::sync(cta);

    if (blockSize == 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        cg::sync(cta);
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        cg::sync(cta);
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        cg::sync(cta);
    }
    if (blockSize >= 128) {
        if (tid < 64)   sdata[tid] += sdata[tid + 64];
        cg::sync(cta);
    }
    if (tid < 32) {
        warpReduce2<T, blockSize>(sdata, tid);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T>
__global__ void reduce5_1(T* g_idata, T* g_odata, unsigned int n) {
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ T sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    if (cta.thread_rank() < 32) {
        // Fetch final intermeddiate sum from 2nd warp
        if (blockDim.x >= 64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            mySum += tile32.shfl_down(mySum, offset);
        }
    }

    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
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
      break;

    case 51:
        reduce5_1<T> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

    case 5:
        switch (threads) {
        case 1024:
            reduce5<T, 1024> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 512:
            reduce5<T, 512> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 256:
            reduce5<T, 256> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 128:
            reduce5<T, 128> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 64:
            reduce5<T, 64> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 32:
            reduce5<T, 32> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 16:
            reduce5<T, 16> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 8:
            reduce5<T, 8> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 4:
            reduce5<T, 4> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 2:
            reduce5<T, 2> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 1:
            reduce5<T, 1> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
      }
      break;
    case 6:
        switch (threads)
        {
        case 1024:
            reduce6<T, 1024> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 512:
            reduce6<T, 512> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 256:
            reduce6<T, 256> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 128:
            reduce6<T, 128> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 64:
            reduce6<T, 64> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 32:
            reduce6<T, 32> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 16:
            reduce6<T, 16> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 8:
            reduce6<T, 8> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 4:
            reduce6<T, 4> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 2:
            reduce6<T, 2> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        case 1:
            reduce6<T, 1> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size); break;
        }
  }
}

template void reduce<float>(int size, int threads, int blocks, int whichKernel, float *d_idata, float *d_odata);