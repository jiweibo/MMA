#pragma once

#include "common/common.h"

class CublasGemm {
public:
  CublasGemm(cudaStream_t stream = nullptr) {
    cublasErrCheck(cublasCreate(&handle_));
    // CUBLAS_TF32_TENSOR_OP_MATH
    cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    if (stream) {
      cublasErrCheck(cublasSetStream(handle_, stream));
    }
  }

  ~CublasGemm() { cublasDestroy(handle_); }

  template <typename T>
  void Gemm(bool trans_a, bool trans_b, int m, int n, int k, T alpha, T beta,
            int lda, int ldb, int ldc, const T *A, const T *B, T *C);

  template <typename T>
  void GemmEx(bool trans_a, bool trans_b, int m, int n, int k, T alpha, T beta,
              int lda, int ldb, int ldc, const T *A, const T *B, T *C);

private:
  CublasGemm(const CublasGemm &) = delete;
  cublasHandle_t handle_;
};

template <>
void CublasGemm::Gemm<float>(bool trans_a, bool trans_b, int m, int n, int k,
                             float alpha, float beta, int lda, int ldb, int ldc,
                             const float *A, const float *B, float *C) {
  cublasErrCheck(cublasSgemm(handle_, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                             trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
                             &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template <>
void CublasGemm::Gemm<half>(bool trans_a, bool trans_b, int m, int n, int k,
                            half alpha, half beta, int lda, int ldb, int ldc,
                            const half *A, const half *B, half *C) {
  cublasErrCheck(cublasHgemm(handle_, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                             trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
                             &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template <>
void CublasGemm::GemmEx<float>(bool trans_a, bool trans_b, int m, int n, int k,
                               float alpha, float beta, int lda, int ldb,
                               int ldc, const float *A, const float *B,
                               float *C) {
  cublasErrCheck(cublasGemmEx(
      handle_, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
      trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_32F, k, B,
      CUDA_R_32F, n, &beta, C, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void CublasGemm::GemmEx<half>(bool trans_a, bool trans_b, int m, int n, int k,
                              half alpha, half beta, int lda, int ldb, int ldc,
                              const half *A, const half *B, half *C) {
  nv_bfloat16 s;
  cublasErrCheck(cublasGemmEx(
      handle_, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
      trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_16F, k, B,
      CUDA_R_16F, n, &beta, C, CUDA_R_16F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void CublasGemm::GemmEx<nv_bfloat16>(bool trans_a, bool trans_b, int m, int n,
                                     int k, nv_bfloat16 alpha, nv_bfloat16 beta,
                                     int lda, int ldb, int ldc,
                                     const nv_bfloat16 *A, const nv_bfloat16 *B,
                                     nv_bfloat16 *C) {
  cublasErrCheck(cublasGemmEx(handle_, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                              trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
                              &alpha, A, CUDA_R_16BF, k, B, CUDA_R_16BF, n,
                              &beta, C, CUDA_R_16BF, n, CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT));
}
