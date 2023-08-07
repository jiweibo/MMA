#pragma once

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <stdio.h>

#define cudaErrCheck(stat)                                                     \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }
static void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file,
            line);
  }
}

#define cublasErrCheck(stat)                                                   \
  { cublasErrCheck_((stat), __FILE__, __LINE__); }
static void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %s %s %d\n", stat, file, line);
  }
}

#define curandErrCheck(stat)                                                   \
  { curandErrCheck_((stat), __FILE__, __LINE__); }
static void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}

// Error check.
template <typename T>
static int HostErrCheck(T *a, T *b, int num, float atol = 1e-5,
                        float rtol = 0.001) {
  int errors = 0;
  for (int i = 0; i < num; ++i) {
    float v1 = a[i];
    float v2 = b[i];
    if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-5) {
      errors++;
    }
  }
  return errors;
}

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *msg, const char *file,
                               const int line) {
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error %s : (%d) %s.\n",
            file, line, msg, static_cast<int>(error),
            cudaGetErrorString(error));
    exit(-1);
  }
}