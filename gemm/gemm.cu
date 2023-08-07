#include <iostream>

#include <cstdlib>
#include <mma.h>
#include <common/common.h>

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(half* a, half* b, float* c, int M, int N, int K, float alpha, float beta) {
  int lda = M;
  int ldb = K;
  int ldc = M;

  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = blockIdx.y * blockDim.y + threadIdx.y;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  nvcuda::wmma::fill_fragment(acc_frag, 0.f);

  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * WMMA_N;
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      nvcuda::wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result scaled by alpha.
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;

  if (cRow < M && cCol < N) {
    nvcuda::wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, nvcuda::wmma::mem_col_major);
    
    for (int i = 0; i < c_frag.num_elements; ++i) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    nvcuda::wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, nvcuda::wmma::mem_col_major);
  }
}

__global__ void ConvertFp32ToFp16(half* out, float* in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

int main(int argc, char** argv) {
  const int MATRIX_M = 1024; //16384;
  const int MATRIX_N = 1024; //16384;
  const int MATRIX_K = 1024; //16384;

  const int warmup = 10;
  const int repeats = 100;

  float alpha = 2.0f;
  float beta = 2.0f;

  float* a_fp32;
  float* b_fp32;
  half* a_fp16;
  half* b_fp16;

  float* c;
  float* c_cublas;
  float* c_wmma;

  float* c_host_cublas;
  float* c_host_wmma;

  curandGenerator_t gen;
  cublasHandle_t cublas_handle;

  cudaEvent_t start_wmma;
  cudaEvent_t stop_wmma;

  cudaEvent_t start_cublas;
  cudaEvent_t stop_cublas;

  cudaErrCheck(cudaEventCreate(&start_wmma));
  cudaErrCheck(cudaEventCreate(&stop_wmma));

  cudaErrCheck(cudaEventCreate(&start_cublas));
  cudaErrCheck(cudaEventCreate(&stop_cublas));

  cublasErrCheck(cublasCreate(&cublas_handle));
  // Use tensor cores
  cublasErrCheck(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

  cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

  cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

  c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

  curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
  curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
  curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

  ConvertFp32ToFp16<<<(MATRIX_M*MATRIX_K+255)/256, 256>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  ConvertFp32ToFp16<<<(MATRIX_M*MATRIX_K+255)/256, 256>>>(b_fp16, b_fp32, MATRIX_K * MATRIX_N);

  curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
  curandErrCheck(curandDestroyGenerator(gen));

  // wmma
  dim3 grid_dim;
  dim3 block_dim;

  // block_dim.x must be a multiple of warp_size;
  // 128 x 4 means we have 16 warps and a block computes a 64x64 output tile.
  block_dim.x = 128;
  block_dim.y = 4;
  grid_dim.x = (MATRIX_M + (WMMA_M * block_dim.x / 32 - 1)) / (WMMA_M * block_dim.x / 32);
  grid_dim.y = (MATRIX_N + (WMMA_N * block_dim.y - 1)) / (WMMA_N * block_dim.y);
  std::cout << "Running with wmma..." << std::endl;
  for (int i = 0; i < warmup; ++i) {
    wmma_example<<<grid_dim, block_dim>>>(a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  }
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaEventRecord(start_wmma));
  for (int i = 0; i < repeats; ++i) {
    wmma_example<<<grid_dim, block_dim>>>(a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  }
  cudaErrCheck(cudaEventRecord(stop_wmma));


  // cublas
  std::cout << "Running with cuBLAS..." << std::endl;
  for (int i = 0; i < warmup; ++i) {
    cublasErrCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, a_fp16, CUDA_R_16F, MATRIX_M,b_fp16, CUDA_R_16F, MATRIX_K, &beta, c_cublas, CUDA_R_32F, MATRIX_M, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));  
  }
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaEventRecord(start_cublas));
  for (int i = 0; i < repeats; ++i) {
    cublasErrCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, a_fp16, CUDA_R_16F, MATRIX_M,b_fp16, CUDA_R_16F, MATRIX_K, &beta, c_cublas, CUDA_R_32F, MATRIX_M, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  cudaErrCheck(cudaEventRecord(stop_cublas));

  // Error Checking
  std::cout << "Checking results..." << std::endl;
  cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

  auto errors = HostErrCheck(c_host_wmma, c_host_cublas, MATRIX_M * MATRIX_N);
  if (errors > 0) {
    std::cout << "WMMA does not agree with cuBLAS! " << errors << " errors." << std::endl;
  } else {
    std::cout << "Result verified: cublas and WMMA agree." << std::endl;
    float cublas_time;
    float wmma_time;
    cudaErrCheck(cudaEventSynchronize(stop_wmma));
    cudaErrCheck(cudaEventSynchronize(stop_cublas));
    cudaErrCheck(cudaEventElapsedTime(&wmma_time, start_wmma, stop_wmma));
    cudaErrCheck(cudaEventElapsedTime(&cublas_time, start_cublas, stop_cublas));
    std::cout << "cublas took " << cublas_time / repeats << " ms, " << 2.f * MATRIX_M * MATRIX_N * MATRIX_K / (cublas_time / repeats) / 1e6 << "GFLOPS" << std::endl;
    std::cout << "wmma took " << wmma_time / repeats << " ms, " << 2.f * MATRIX_M * MATRIX_N * MATRIX_K / (wmma_time / repeats) / 1e6 << "GFLOPS" << std::endl;
  }

  cudaErrCheck(cudaEventDestroy(start_wmma));
  cudaErrCheck(cudaEventDestroy(stop_wmma));
  cudaErrCheck(cudaEventDestroy(start_cublas));
  cudaErrCheck(cudaEventDestroy(stop_cublas));

  cudaErrCheck(cudaFree(a_fp32));
  cudaErrCheck(cudaFree(b_fp32));
  cudaErrCheck(cudaFree(a_fp16));
  cudaErrCheck(cudaFree(b_fp16));
  cudaErrCheck(cudaFree(c));
  cudaErrCheck(cudaFree(c_cublas));
  cudaErrCheck(cudaFree(c_wmma));
  free(c_host_cublas);
  free(c_host_wmma);
}