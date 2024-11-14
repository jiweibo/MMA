#pragma once

#include <cuda_runtime_api.h>

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
             Beta beta, TC *C, int ldC, cudaStream_t stream = 0);

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
             Beta beta, TC *C, int ldC, cudaStream_t stream = 0);

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_mnm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream = nullptr);

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_mkm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream = nullptr);

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_knm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream = nullptr);

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_kkm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream = nullptr);

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt2(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream = nullptr);

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn2(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream = nullptr);