#include "common/common.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 1;

// Check erros and print GB/s
void PostProcess(const float *ref, const float *res, int n, float ms) {
  bool passed = true;
  for (int i = 0; i < n; ++i) {
    if (res[i] != ref[i]) {
      std::cout << i << " " << res[i] << " " << ref[i] << std::endl;
      passed = false;
      break;
    }
  }
  if (passed) {
    std::cout << std::setiosflags(std::ios::left) << std::setw(25)
              << 2 * n * sizeof(float) * NUM_REPS * 1e-6 / ms << std::endl;
    ;
  }
}

// simple copy kernel
// Used as reference case representing best effective bandwidth.
template <typename T> __global__ void Copy(T *out_data, const T *in_data) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    out_data[(y + j) * width + x] = in_data[(y + j) * width + x];
  }
}

// copy kernel using shared memory
// Also used as reference case, demonstrating effect of using shared memory.
template <typename T>
__global__ void CopySharedMem(T *out_data, const T *in_data) {
  __shared__ T tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile[threadIdx.y + j][threadIdx.x] = in_data[(y + j) * width + x];
  }

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    out_data[(y + j) * width + x] = tile[(threadIdx.y + j)][threadIdx.x];
  }
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
template <typename T>
__global__ void TransposeNaive(T *out_data, const T *in_data) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    out_data[x * width + (y + j)] = in_data[(y + j) * width + x];
  }
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
template <typename T>
__global__ void TransposeCoalesced(T *out_data, const T *in_data) {
  __shared__ T tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile[threadIdx.y + j][threadIdx.x] = in_data[(y + j) * width + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    out_data[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

// No bank-conflict transpose
// Same as TransposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
template <typename T>
__global__ void TransposeNoBankConflicts(T *out_data, const T *in_data) {
  __shared__ T tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile[threadIdx.y + j][threadIdx.x] = in_data[(y + j) * width + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    out_data[(y + j) * width + x] = tile[threadIdx.x][j + threadIdx.y];
  }
}

int main(int argc, char **argv) {
  const int nx = 1024;
  const int ny = 1024;
  const int mem_size = nx * ny * sizeof(float);

  dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int dev_id = 0;
  if (argc > 1)
    dev_id = atoi(argv[1]);

  cudaDeviceProp prop;
  cudaErrCheck(cudaGetDeviceProperties(&prop, dev_id));
  std::cout << "Device : " << prop.name << std::endl;
  std::cout << "Matrix size: " << nx << " " << ny
            << ", Block size: " << TILE_DIM << " " << TILE_DIM
            << ", Tile size: " << TILE_DIM << " " << BLOCK_ROWS << std::endl;
  std::cout << "dimGrid: " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z
            << ". dimBlock: " << dimBlock.x << " " << dimBlock.y << " "
            << dimBlock.z << std::endl;
  cudaErrCheck(cudaSetDevice(dev_id));

  float *h_idata = (float *)malloc(mem_size);
  float *h_cdata = (float *)malloc(mem_size);
  float *h_tdata = (float *)malloc(mem_size);
  float *gold = (float *)malloc(mem_size);

  float *d_idata, *d_cdata, *d_tdata;
  cudaErrCheck(cudaMalloc(&d_idata, mem_size));
  cudaErrCheck(cudaMalloc(&d_cdata, mem_size));
  cudaErrCheck(cudaMalloc(&d_tdata, mem_size));

  if (nx % TILE_DIM || ny % TILE_DIM) {
    std::cout << "nx and ny must be multiple of TILE_DIM" << std::endl;
    goto error_exit;
  }
  if (TILE_DIM % BLOCK_ROWS) {
    std::cout << "TILE_DIM must be multiple of BLOCK_ROWS" << std::endl;
    goto error_exit;
  }

  // host
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      h_idata[j * nx + i] = j * nx + i;
    }
  }

  // correct result for error checking
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      gold[j * nx + i] = h_idata[i * nx + j];
    }
  }

  cudaErrCheck(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

  // event for timing
  cudaEvent_t start_event, stop_event;
  cudaErrCheck(cudaEventCreate(&start_event));
  cudaErrCheck(cudaEventCreate(&stop_event));
  float ms;

  // -------------
  // time kernels
  // -------------
  std::cout << std::setiosflags(std::ios::left) << std::setw(25) << "Routine"
            << std::setw(25) << "Bandwidth(GB/s)" << std::endl;

  // ----
  // copy
  // ----
  std::cout << std::setiosflags(std::ios::left) << std::setw(25) << "Copy";
  cudaErrCheck(cudaMemset(d_cdata, 0, mem_size));
  Copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  cudaErrCheck(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i) {
    Copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  }
  cudaErrCheck(cudaEventRecord(stop_event, 0));
  cudaErrCheck(cudaEventSynchronize(stop_event));
  cudaErrCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
  cudaErrCheck(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
  PostProcess(h_idata, h_cdata, nx * ny, ms);

  // ----------------
  // CopySharedMem
  // ----------------
  std::cout << std::setiosflags(std::ios::left) << std::setw(25)
            << "CopySharedMem";
  cudaErrCheck(cudaMemset(d_cdata, 0, mem_size));
  CopySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  cudaErrCheck(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i) {
    CopySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  }
  cudaErrCheck(cudaEventRecord(stop_event, 0));
  cudaErrCheck(cudaEventSynchronize(stop_event));
  cudaErrCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
  cudaErrCheck(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
  PostProcess(h_idata, h_cdata, nx * ny, ms);

  // ----------------
  // TransposeNaive
  // ----------------
  std::cout << std::setiosflags(std::ios::left) << std::setw(25)
            << "TransposeNaive";
  cudaErrCheck(cudaMemset(d_tdata, 0, mem_size));
  // warmup
  TransposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  cudaErrCheck(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i) {
    TransposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  }
  cudaErrCheck(cudaEventRecord(stop_event, 0));
  cudaErrCheck(cudaEventSynchronize(stop_event));
  cudaErrCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
  cudaErrCheck(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  PostProcess(gold, h_tdata, nx * ny, ms);

  // ----------------
  // TransposeColasced
  // ----------------
  std::cout << std::setiosflags(std::ios::left) << std::setw(25)
            << "TransposeColasced";
  cudaErrCheck(cudaMemset(d_tdata, 0, mem_size));
  // warmup
  TransposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  cudaErrCheck(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i) {
    TransposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  }
  cudaErrCheck(cudaEventRecord(stop_event, 0));
  cudaErrCheck(cudaEventSynchronize(stop_event));
  cudaErrCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
  cudaErrCheck(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  PostProcess(gold, h_tdata, nx * ny, ms);

  // ----------------
  // TransposeNoBankConflicts
  // ----------------
  std::cout << std::setiosflags(std::ios::left) << std::setw(25)
            << "TransposeNoBankConflicts";
  cudaErrCheck(cudaMemset(d_tdata, 0, mem_size));
  // warmup
  TransposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  cudaErrCheck(cudaEventRecord(start_event, 0));
  for (int i = 0; i < NUM_REPS; ++i) {
    TransposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  }
  cudaErrCheck(cudaEventRecord(stop_event, 0));
  cudaErrCheck(cudaEventSynchronize(stop_event));
  cudaErrCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
  cudaErrCheck(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  PostProcess(gold, h_tdata, nx * ny, ms);

error_exit:
  // clean up
  cudaErrCheck(cudaFree(d_idata));
  cudaErrCheck(cudaFree(d_cdata));
  cudaErrCheck(cudaFree(d_tdata));
  free(h_idata);
  free(h_cdata);
  free(h_tdata);
  free(gold);
}
