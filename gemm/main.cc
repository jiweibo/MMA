#include "common/common.h"
#include "common/timer.h"

#include <cstring>
#include <random>
#include <string>

#include "gemm/cublas_gemm.h"
#include "gemm/cublaslt_gemm.h"
#include "gemm/cutlass_gemm.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_int32(m, 128, "m");
DEFINE_int32(n, 128, "n");
DEFINE_int32(k, 128, "k");
DEFINE_bool(check_precision, false, "check value error.");

namespace {

std::default_random_engine e(1998);
std::uniform_real_distribution<float> u(-1, 1);

void GenerateRandomData(float *data, int num) {
  for (int i = 0; i < num; ++i) {
    data[i] = u(e);
  }
}

// RowMajoir C = alpha * A*B + beta * C
void SgemmRef(int M, int N, int K, float alpha, float beta, int lda, int ldb, int ldc,
              const float *A, const float *B, float *C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float prod = 0.f;
      for (int k = 0; k < K; ++k) {
        prod += A[m * K + k] * B[k * N + n];
      }

      C[m * N + n] = alpha * prod + beta * C[m * N + n];
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  using DTYPE = float;

  int M = FLAGS_m;
  int N = FLAGS_n;
  int K = FLAGS_k;

  int a_size = M * K;
  int b_size = K * N;
  int c_size = M * N;

  DTYPE *host_a = (DTYPE *)malloc(a_size * sizeof(DTYPE));
  DTYPE *host_b = (DTYPE *)malloc(b_size * sizeof(DTYPE));
  DTYPE *host_c = (DTYPE *)malloc(c_size * sizeof(DTYPE));
  DTYPE *host_c_ref = (DTYPE *)malloc(c_size * sizeof(DTYPE));
  memset(host_c_ref, 0, c_size);

  DTYPE *device_a;
  DTYPE *device_b;
  DTYPE *device_c;
  float alpha = 1.0f;
  float beta = 0.f;

  cudaStream_t stream;
  cudaErrCheck(cudaStreamCreate(&stream));

  GenerateRandomData(host_a, a_size);
  GenerateRandomData(host_b, b_size);

  SgemmRef(M, N, K, alpha, beta, K, N, N, host_a, host_b, host_c_ref);

  cudaErrCheck(cudaMalloc((void **)&device_a, a_size * sizeof(DTYPE)));
  cudaErrCheck(cudaMalloc((void **)&device_b, b_size * sizeof(DTYPE)));
  cudaErrCheck(cudaMalloc((void **)&device_c, c_size * sizeof(DTYPE)));
  cudaMemset(device_c, 0, c_size * sizeof(DTYPE));
  cudaErrCheck(cudaMemcpy(device_a, host_a, a_size * sizeof(DTYPE), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(device_b, host_b, b_size * sizeof(DTYPE), cudaMemcpyHostToDevice));

  // CublasGemm gemm(stream);
  // if (FLAGS_check_precision) {
  //   LOG(INFO) << "Check Precision for CUBLAS Sgemm";
  //   gemm.Gemm(false, false, N, M, K, alpha, beta, N, K, N, device_b, device_a, device_c);
  //   cudaErrCheck(
  //       cudaMemcpyAsync(host_c, device_c, c_size * sizeof(DTYPE), cudaMemcpyDeviceToHost,
  //       stream));
  //   cudaErrCheck(cudaStreamSynchronize(stream));
  //   LOG(INFO) << "Err: " << HostErrCheck(host_c_ref, host_c, c_size);
  // }
  // BenchFunc(FLAGS_warmup, FLAGS_repeats, "blas.gemm", [&]() {
  //   gemm.Gemm(false, false, N, M, K, alpha, beta, N, K, N, device_b, device_a, device_c);
  // });

  // CublasLtGemm ltgemm(16 * 1024 * 1024);
  // ltgemm.Gemm(false, false, M, N, K, alpha, beta, K, N, N, device_a,
  // device_b,
  //             device_c);
  // cudaErrCheck(cudaMemcpy(host_c, device_c, c_size * sizeof(DTYPE),
  //                         cudaMemcpyDeviceToHost));
  // LOG(INFO) << "Err: " << HostErrCheck(host_c_ref, host_c, c_size);

  // BenchFunc(FLAGS_warmup, FLAGS_repeats, "blas.gemm", [&]() {
  //   gemm.GemmEx(false, false, M, N, K, alpha, beta, K, N, N, device_a,
  //   device_b,
  //               device_c);
  // });

  // BenchFunc(FLAGS_warmup, FLAGS_repeats, "blaslt.gemm", [&]() {
  //   ltgemm.Gemm(false, false, M, N, K, alpha, beta, K, N, N, device_a,
  //   device_b,
  //               device_c);
  // });

  // if (FLAGS_check_precision) {
  //   LOG(INFO) << "Check Precision for CUTLASS gemm_mkm";
  //   gemm_mkm(N, M, K, alpha, device_b, N, device_a, K, beta, device_c, N,
  //            stream);
  //   cudaErrCheck(cudaMemcpyAsync(host_c, device_c, c_size * sizeof(DTYPE),
  //                                cudaMemcpyDeviceToHost, stream));
  //   cudaErrCheck(cudaStreamSynchronize(stream));
  //   LOG(INFO) << "Err: " << HostErrCheck(host_c_ref, host_c, c_size);
  // }
  // BenchFunc(FLAGS_warmup, FLAGS_repeats, "gemm_mkm", [&]() {
  //   gemm_mkm(N, M, K, alpha, device_b, N, device_a, K, beta, device_c, N,
  //            stream);
  //   cudaErrCheck(cudaStreamSynchronize(stream));
  // });

  // A: NxK:1xldb KxN:ldbx1
  // B: MxK:ldax1
  // C: NxM:1xldc
  DTYPE *host_b_trans = (DTYPE *)malloc(b_size * sizeof(DTYPE));
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < K; ++j) {
      host_b_trans[i * K + j] = host_b[j * N + i];
    }
  }
  cudaErrCheck(cudaMalloc((void **)&device_b, b_size * sizeof(DTYPE)));
  cudaErrCheck(cudaMemcpy(device_b, host_b_trans, b_size * sizeof(DTYPE), cudaMemcpyHostToDevice));
  gemm_tn2(N, M, K, alpha, device_b, K, device_a, K, beta, device_c, N);
  cudaErrCheck(
      cudaMemcpyAsync(host_c, device_c, c_size * sizeof(DTYPE), cudaMemcpyDeviceToHost, stream));
  cudaErrCheck(cudaStreamSynchronize(stream));
  LOG(INFO) << "Err: " << HostErrCheck(host_c_ref, host_c, c_size);
  BenchFunc(FLAGS_warmup, FLAGS_repeats, "gemm_tn2", [&]() {
    gemm_tn2(N, M, K, alpha, device_b, K, device_a, K, beta, device_c, N, stream);
    cudaErrCheck(cudaStreamSynchronize(stream));
  });

  cudaErrCheck(cudaFree(device_a));
  cudaErrCheck(cudaFree(device_b));
  cudaErrCheck(cudaFree(device_c));
  free(host_a);
  free(host_b);
  free(host_c);
  free(host_c_ref);

  return 0;
}