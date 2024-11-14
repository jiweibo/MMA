#pragma once

#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <iostream>
#include <stdio.h>

#include <glog/logging.h>

#include "common/timer.h"

namespace {
inline const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "Unknown cuBLAS error";
  }
}
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file,
            line);
  }
}

inline void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %s %s %d\n", cublasGetErrorString(stat),
            file, line);
  }
}

inline void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}

inline void __getLastCudaError(const char *msg, const char *file,
                               const int line) {
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error %s : (%d) %s.\n",
            file, line, msg, static_cast<int>(error),
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}
} // namespace

#define cudaErrCheck(stat)                                                     \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }

#define cublasErrCheck(stat)                                                   \
  { cublasErrCheck_((stat), __FILE__, __LINE__); }

#define curandErrCheck(stat)                                                   \
  { curandErrCheck_((stat), __FILE__, __LINE__); }

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

// Error check.
template <typename T>
inline int HostErrCheck(T *a, T *b, int num, float atol = 1e-5,
                        float rtol = 0.001) {
  int errors = 0;
  for (int i = 0; i < num; ++i) {
    float abs_err = std::abs(a[i] - b[i]);
    float rel_err = std::abs(a[i] / b[i] - 1.0f);
    if (abs_err > atol && rel_err > rtol) {
      std::cout << "Error at index " << i << ": " << a[i] << " != " << b[i]
                << ", abs_error = " << abs_err
                << ", relative_error = " << rel_err << std::endl;
      errors++;
    }
  }
  return errors;
}

inline void BenchFunc(int warmup, int repeats, const std::string &info,
                      const std::function<void()> &func) {
  // warmup
  for (int i = 0; i < warmup; ++i) {
    func();
  }
  cudaDeviceSynchronize();

  StopWatchTimer timer;
  for (int i = 0; i < repeats; ++i) {
    timer.Start();
    func();
    timer.Stop();
  }

  LOG(INFO) << info << ", Average time: " << timer.GetAverageTime()
            << ", percentile(50): " << timer.ComputePercentile(0.5)
            << ", percentile(90): " << timer.ComputePercentile(0.9)
            << ", percentile(99): " << timer.ComputePercentile(0.99)
            << ", variance: " << timer.ComputeVariance();
}
