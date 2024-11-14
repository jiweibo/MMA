#pragma once

#include "common/common.h"

#include "common/lru_cache.h"
#include "cute/container/bit_field.hpp"

#include <cstdint>
#include <vector>

#include <cublasLt.h>

namespace {

struct BlasLtKeyParams {
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
  bool trans_a;
  bool trans_b;
};

bool operator==(const BlasLtKeyParams &lhs, const BlasLtKeyParams &rhs) {
  return lhs.m == rhs.m && lhs.n == rhs.n && lhs.k == rhs.k &&
         lhs.lda == rhs.lda && lhs.ldb == rhs.ldb && lhs.ldc == rhs.ldc &&
         lhs.trans_a == rhs.trans_a && lhs.trans_b == rhs.trans_b;
}

struct BlasLtValueParams {
  cublasLtMatmulDesc_t operation_desc;
  cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
  cublasLtMatmulPreference_t preference;
  cublasOperation_t a_op, b_op;
  cublasLtMatmulAlgo_t algo;

  ~BlasLtValueParams() {
    if (preference) {
      cublasLtMatmulPreferenceDestroy(preference);
    }
    if (c_desc) {
      cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc) {
      cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc) {
      cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (operation_desc) {
      cublasLtMatmulDescDestroy(operation_desc);
    }
  }
};

// Structure to store information about difference run trials
struct customMatmulPerf_t {
  cublasLtMatmulAlgo_t algo;
  cublasStatus_t status;
  float time;
  size_t workspace_size; // actual memory workspace needed
  cublasMath_t math_mode;
  cublasLtReductionScheme_t reduction_scheme;
  int custom_option;
  float waves_count;

  cublasLtMatmulDesc_t operation_desc;
  cublasLtMatrixLayout_t a_desc;
  cublasLtMatrixLayout_t b_desc;
  cublasLtMatrixLayout_t c_desc;
};

static inline bool TimeCompare(const customMatmulPerf_t &perf_a,
                               const customMatmulPerf_t &perf_b) {
  return (perf_a.status == CUBLAS_STATUS_SUCCESS) &&
         (perf_a.time < perf_b.time);
}

// float median(std::vector<float> &times) {
//   const size_t size = times.size();
//   if (size == 0) {
//     return 0;
//   }

//   std::sort(times.begin(), times.end());

//   const size_t mid = size / 2;
//   if (size % 2 == 0) {
//     return (times[mid] + times[mid - 1]) / 2;
//   } else {
//     return times[mid];
//   }
// }
} // namespace

namespace std {
template <> struct hash<BlasLtKeyParams> {
  std::size_t operator()(const BlasLtKeyParams &params) const noexcept {
    std::size_t seed = 1998;
    hash_combine(seed, params.m, params.n, params.k, params.lda, params.ldb,
                 params.ldc, params.trans_a, params.trans_b);
    return seed;
  }
};
} // namespace std

/* CAUTION : must match cublasLtMatmulTile_t */
const char *const matmulTileName[] = {
    "UNDEF",  "8x8",    "8x16",    "16x8",    "8x32",   "16x16",  "32x8",
    "8x64",   "16x32",  "32x16",   "64x8",    "32x32",  "32x64",  "64x32",
    "32x128", "64x64",  "128x32",  "64x128",  "128x64", "64x256", "128x128",
    "256x64", "64x512", "128x256", "256x128", "512x64",
};

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure(const customMatmulPerf_t &perf) {
  int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;

  const cublasLtMatmulAlgo_t *matmulAlgo = &perf.algo;
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID,
                                       &algoId, sizeof(algoId), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                       &tile, sizeof(tile), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo,
                                       CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                       &numSplitsK, sizeof(numSplitsK), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(
      matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme,
      sizeof(reductionScheme), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo,
                                       CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                       &swizzle, sizeof(swizzle), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(
      matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption,
      sizeof(customOption), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo,
                                       CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages,
                                       sizeof(stages), NULL);

  printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d "
         "custom=%d stages=%d} status %d "
         "time %f workspace=%d mathMode=%d waves=%f\n",
         algoId, tile, matmulTileName[tile], numSplitsK, reductionScheme,
         swizzle, customOption, stages, perf.status, perf.time,
         (int)perf.workspace_size, (int)perf.math_mode, perf.waves_count);
}

#define ALGO_COMBINATIONS 10000
#define ALGO_IDS 100

// References: https://docs.nvidia.com/cuda/cublas/#cublaslt-code-examples
class CublasLtGemm {
public:
  CublasLtGemm(size_t workspace_size = 4 * 1024 * 1024,
               size_t max_cache_size = 1000)
      : workspace_size_(workspace_size), caches_(max_cache_size) {
    cudaErrCheck(cudaStreamCreate(&stream_));
    cublasErrCheck(cublasLtCreate(&hanlde_));
    cudaErrCheck(cudaMalloc(&workspace_, workspace_size));
    cudaErrCheck(cudaEventCreate(&start_event_, cudaEventBlockingSync));
    cudaErrCheck(cudaEventCreate(&stop_event_, cudaEventBlockingSync));
  }

  ~CublasLtGemm() {
    cudaFree(workspace_);
    cublasLtDestroy(hanlde_);
    cudaStreamDestroy(stream_);
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
  }

  template <typename T>
  void Gemm(bool trans_a, bool trans_b, int m, int n, int k, T alpha, T beta,
            int lda, int ldb, int ldc, const T *A, const T *B, T *C);

private:
  cublasStatus_t CustomMatmulRun(
      cublasLtHandle_t handle, cublasLtMatmulDesc_t operation_desc,
      const void *alpha, const void *A, cublasLtMatrixLayout_t a_desc,
      const void *B, cublasLtMatrixLayout_t b_desc, const void *beta,
      const void *C, cublasLtMatrixLayout_t c_desc, void *D,
      cublasLtMatrixLayout_t d_desc, const cublasLtMatmulAlgo_t &algo,
      int kernel_epeats, void *workspace, size_t workspace_size_in_bytes,
      customMatmulPerf_t &perfResults, cudaStream_t stream,
      cudaEvent_t start_event, cudaEvent_t stop_event) {
    cublasLtMatmulHeuristicResult_t heur_result;
    // Looping over the Algo
    int repeats = kernel_epeats;

    cublasStatus_t algo_status =
        cublasLtMatmulAlgoCheck(handle, operation_desc, a_desc, b_desc, c_desc,
                                d_desc, &algo, &heur_result);

    if (algo_status == CUBLAS_STATUS_SUCCESS) {
      if (heur_result.workspaceSize <= workspace_size_in_bytes) {
        cudaError_t err, err1, err2, err3;
        err = cudaEventRecord(start_event, stream);
        for (int loop = 0; loop < repeats; ++loop) {
          cublasStatus_t one_run_status =
              cublasLtMatmul(handle, operation_desc, alpha, A, a_desc, B,
                             b_desc, beta, C, c_desc, D, d_desc, &algo,
                             workspace, workspace_size_in_bytes, stream);
          if (one_run_status != CUBLAS_STATUS_SUCCESS) {
            algo_status = one_run_status;
            break;
          }
        }
        err1 = cudaEventRecord(stop_event, stream);
        err2 = cudaEventSynchronize(stop_event);
        float time;
        err3 = cudaEventElapsedTime(&time, start_event, stop_event);
        if ((err != cudaSuccess) || (err1 != cudaSuccess) ||
            (err2 != cudaSuccess) || (err3 != cudaSuccess)) {
          algo_status = CUBLAS_STATUS_INTERNAL_ERROR;
        }
        // For the moment only add successful findings
        if (algo_status == CUBLAS_STATUS_SUCCESS) {
          perfResults.algo = algo;
          perfResults.time = time;
          perfResults.workspace_size = heur_result.workspaceSize;
          perfResults.waves_count = heur_result.wavesCount;
        }
      } else {
        algo_status = CUBLAS_STATUS_NOT_SUPPORTED; // Not enough workspace
      }
    }
    return algo_status;
  }

  // Sample wrapper running through multiple algo and config attributes
  // combination for single precision gemm using cublasLt lower-level API
  void LtSgemmCustomFind(cublasLtHandle_t handle, cublasOperation_t trans_a,
                         cublasOperation_t trans_b, int m, int n, int k,
                         const float *alpha, const float *A, int lda,
                         const float *B, int ldb, const float *beta, float *C,
                         int ldc, void *workspace, size_t workspace_size,
                         cudaStream_t stream, cudaEvent_t start_event,
                         cudaEvent_t stop_event) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulDesc_t operation_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
    // let try a fixed number of combinations
    int AlgoCombinations = ALGO_COMBINATIONS;
    int AlgoCount = 0;
    // number of time the CUDA kernels will be run back to back
    int kernel_repeats = 10;
    customMatmulPerf_t perfResults[AlgoCombinations];
    int nbAlgoIds = 0;
    int algoIdA[ALGO_IDS];
    cudaDataType_t scale_type = CUDA_R_32F, a_type = CUDA_R_32F,
                   b_type = CUDA_R_32F, c_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; here we just need to
    // set the transforms for A and B
    cublasErrCheck(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F,
                                            CUDA_R_32F));
    cublasErrCheck(cublasLtMatmulDescSetAttribute(operation_desc,
                                                  CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &trans_a, sizeof(trans_a)));
    cublasErrCheck(cublasLtMatmulDescSetAttribute(operation_desc,
                                                  CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &trans_b, sizeof(trans_b)));

    // create matrix descriptors, we are good with the details here so no
    // need to set any extra attributes
    cublasErrCheck(cublasLtMatrixLayoutCreate(
        &a_desc, CUDA_R_32F, trans_a == CUBLAS_OP_N ? m : k,
        trans_a == CUBLAS_OP_N ? k : m, lda));
    cublasErrCheck(cublasLtMatrixLayoutCreate(
        &b_desc, CUDA_R_32F, trans_b == CUBLAS_OP_N ? k : n,
        trans_b == CUBLAS_OP_N ? n : k, ldb));
    cublasErrCheck(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m, n, ldc));

    // Request the 4 first algo_id available for SGEMM (compute_type =
    // scale_type = a_type = b_type = c_type = d_type = CUDA_R_32F)
    cublasErrCheck(cublasLtMatmulAlgoGetIds(handle, compute_type, scale_type,
                                            a_type, b_type, c_type, c_type,
                                            ALGO_IDS, algoIdA, &nbAlgoIds));

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations);
         ++idx) {
      cublasLtMatmulAlgo_t algo;
      size_t sizeWritten = 0;

      // Initialize algo structure with given Algo ID
      status =
          cublasLtMatmulAlgoInit(handle, compute_type, scale_type, a_type,
                                 b_type, c_type, c_type, algoIdA[idx], &algo);
      if (status != CUBLAS_STATUS_SUCCESS) {
        continue;
      }

      // Query the tiles enums supported by that algo
      cublasErrCheck(cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten));
      int nbTiles = static_cast<int>(sizeWritten / sizeof(int));
      int *tileA = new int[nbTiles == 0 ? 1 : nbTiles];
      if (nbTiles == 0) {
        tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
        nbTiles = 1;
      }

      cublasErrCheck(cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sizeWritten));
      int nbStages = static_cast<int>(sizeWritten / sizeof(int));
      std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
      if (nbStages == 0) {
        stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
        nbStages = 1;
      } else {
        cublasErrCheck(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stagesA.data(),
            sizeof(int) * nbStages, &sizeWritten));
      }

      int splitkSupport, redMask, swizzlingMax, customOptionMax;
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS,
                                        tileA, sizeof(int) * nbTiles,
                                        &sizeWritten);
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                        &splitkSupport, sizeof(splitkSupport),
                                        &sizeWritten);
      cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask,
          sizeof(redMask), &sizeWritten);
      cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax,
          sizeof(swizzlingMax), &sizeWritten);
      cublasLtMatmulAlgoCapGetAttribute(
          &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax,
          sizeof(customOptionMax), &sizeWritten);

      // Loop over the difference tiles
      for (int tileIdx = 0; tileIdx < nbTiles; ++tileIdx) {
        // Loop over difference stages count
        for (int stagesIdx = 0; stagesIdx < nbStages; ++stagesIdx) {
          cublasErrCheck(cublasLtMatmulAlgoConfigSetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx],
              sizeof(stagesA[stagesIdx])));
          // Loop over the different custom option if any
          for (int customOption = 0; customOption <= customOptionMax;
               ++customOption) {
            cublasErrCheck(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption,
                sizeof(customOption)));
            // loop over the CTAs swizzling support
            for (int k = 0; k <= swizzlingMax; ++k) {
              int splitK_trial = 0;
              if (splitkSupport) {
                splitK_trial +=
                    sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
              }
              // Loop over the splitK value over a fixed sequence
              // splitKSequenceA in addition to the case where splitK is not
              // enabled
              for (int l = 0;
                   (l < 1 + splitK_trial) && (AlgoCount < AlgoCombinations);
                   ++l) {
                // Setup attribute of the algo to run
                cublasErrCheck(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx],
                    sizeof(tileA[tileIdx])));
                int splitK_val = 0;
                int redSchema = CUBLASLT_REDUCTION_SCHEME_NONE;
                cublasErrCheck(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val,
                    sizeof(splitK_val)));
                cublasErrCheck(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
                cublasErrCheck(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redSchema,
                    sizeof(int)));

                if (l > 0) { // Split-K case
                  splitK_val = splitKSequenceA[l - 1];
                  cublasErrCheck(cublasLtMatmulAlgoConfigSetAttribute(
                      &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                      &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1])));
                  // Going over all the reduction scheme
                  for (redSchema = 1;
                       redSchema < CUBLASLT_REDUCTION_SCHEME_MASK &&
                       (AlgoCount < AlgoCombinations);
                       redSchema = redSchema << 1) {
                    if (redSchema & redMask) {
                      cublasErrCheck(cublasLtMatmulAlgoConfigSetAttribute(
                          &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                          &redSchema, sizeof(redSchema)));

                      status = CustomMatmulRun(
                          handle, operation_desc, alpha, A, a_desc, B, b_desc,
                          beta, C, c_desc, C, c_desc, algo, kernel_repeats,
                          workspace, workspace_size, perfResults[AlgoCount],
                          stream, start_event, stop_event);
                      perfResults[AlgoCount].status = status;
                      if (status == CUBLAS_STATUS_SUCCESS)
                        AlgoCount++;
                    }    // end if
                  }      // end for
                } else { // Non-splitK case
                  // if user preference is ok with workspace
                  if (AlgoCount < AlgoCombinations) {
                    status = CustomMatmulRun(
                        handle, operation_desc, alpha, A, a_desc, B, b_desc,
                        beta, C, c_desc, C, c_desc, algo, kernel_repeats,
                        workspace, workspace_size, perfResults[AlgoCount],
                        stream, start_event, stop_event);
                    perfResults[AlgoCount].status = status;
                    if (status == CUBLAS_STATUS_SUCCESS)
                      AlgoCount++;
                  }
                }
              } // end l
            }   // end k
          }     // end customOption
        }       // end stagesIdx
      }         // end tileIdx
      delete[] tileA;
    } // end idx

    // Sort the results per run duration
    std::sort(perfResults, perfResults + AlgoCount, TimeCompare);
    // Print timing and perf details
    for (int i = 0; i < 20; i++) {
      printf("result %03d : ", i);
      printPerfStructure(perfResults[i]);
    }

    BlasLtKeyParams key{m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                        trans_a == CUBLAS_OP_T ? true : false,
                        trans_b == CUBLAS_OP_T ? true : false};
    perfResults[0].operation_desc = operation_desc;
    perfResults[0].a_desc = a_desc;
    perfResults[0].b_desc = b_desc;
    perfResults[0].c_desc = c_desc;
    caches_.Put(key, perfResults[0]);
    // TODO: dtor.
  }

private:
  cudaStream_t stream_;
  cublasLtHandle_t hanlde_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  size_t workspace_size_;
  void *workspace_;

  // TODO(wilber): ThreadSafe.
  // LRUCache<BlasLtKeyParams, BlasLtValueParams> caches_;
  LRUCache<BlasLtKeyParams, customMatmulPerf_t> caches_;
};

template <>
void CublasLtGemm::Gemm<float>(bool trans_a, bool trans_b, int m, int n, int k,
                               float alpha, float beta, int lda, int ldb,
                               int ldc, const float *A, const float *B,
                               float *C) {
  BlasLtKeyParams key{m, n, k, lda, ldb, ldc, trans_a, trans_b};
  if (caches_.Exists(key)) {
    auto &params = caches_.Get(key);
    cublasErrCheck(cublasLtMatmul(hanlde_, params.operation_desc, &alpha, A,
                                  params.a_desc, B, params.b_desc, &beta, C,
                                  params.c_desc, C, params.c_desc, &params.algo,
                                  workspace_, workspace_size_, stream_));
    cudaErrCheck(cudaStreamSynchronize(stream_));
    return;
  }
  LtSgemmCustomFind(hanlde_, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                    trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A,
                    lda, B, ldb, &beta, C, ldc, workspace_, workspace_size_,
                    stream_, start_event_, stop_event_);

  auto &params = caches_.Get(key);
  cublasErrCheck(cublasLtMatmul(hanlde_, params.operation_desc, &alpha, A,
                                params.a_desc, B, params.b_desc, &beta, C,
                                params.c_desc, C, params.c_desc, &params.algo,
                                workspace_, workspace_size_, stream_));
  cudaErrCheck(cudaStreamSynchronize(stream_));

  // ======== higher API ========
  // if (caches_.Exists(key)) {
  //   auto &params = caches_.Get(key);
  //   cublasErrCheck(cublasLtMatmul(hanlde_, params.operation_desc, &alpha, A,
  //                                 params.a_desc, B, params.b_desc, &beta, C,
  //                                 params.c_desc, C, params.c_desc,
  //                                 &params.algo, workspace_, workspace_size_,
  //                                 stream_));
  //   cudaErrCheck(cudaStreamSynchronize(stream_));
  //   return;
  // }
  // cublasLtMatmulDesc_t operation_desc = nullptr;
  // cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc =
  // nullptr; cudaDataType_t scale_type = CUDA_R_32F, a_type = CUDA_R_32F,
  //                b_type = CUDA_R_32F, c_type = CUDA_R_32F;
  // cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  // // create operation desciriptor; see cublasLtMatmulDescAttributes_t for
  // // details about defaults; here we just need to
  // // set the transforms for A and B
  // cublasErrCheck(cublasLtMatmulDescCreate(&operation_desc,
  // CUBLAS_COMPUTE_32F,
  //                                         CUDA_R_32F));
  // cublasErrCheck(cublasLtMatmulDescSetAttribute(
  //     operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a,
  //     sizeof(trans_a)));
  // cublasErrCheck(cublasLtMatmulDescSetAttribute(
  //     operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b,
  //     sizeof(trans_b)));

  // // create matrix descriptors, we are good with the details here so no need
  // // to set any extra attributes
  // cublasErrCheck(cublasLtMatrixLayoutCreate(
  //     &a_desc, CUDA_R_32F, trans_a == CUBLAS_OP_N ? m : k,
  //     trans_a == CUBLAS_OP_N ? k : m, lda));
  // cublasErrCheck(cublasLtMatrixLayoutCreate(
  //     &b_desc, CUDA_R_32F, trans_b == CUBLAS_OP_N ? k : n,
  //     trans_b == CUBLAS_OP_N ? n : k, ldb));
  // cublasErrCheck(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m, n, ldc));

  // customMatmulPerf_t perfResults;

  // LtSgemmCustomFind(hanlde_, trans_a == false ? CUBLAS_OP_N : CUBLAS_OP_T,
  //                   trans_b == false ? CUBLAS_OP_N : CUBLAS_OP_T, m, n, k,
  //                   &alpha, A, lda, B, ldb, &beta, C, ldc, workspace_,
  //                   workspace_size_);

  // if (caches_.Exists(key)) {
  //   auto &params = caches_.Get(key);
  //   cublasErrCheck(cublasLtMatmul(hanlde_, params.operation_desc, &alpha, A,
  //                                 params.a_desc, B, params.b_desc, &beta, C,
  //                                 params.c_desc, C, params.c_desc,
  //                                 &params.algo, workspace_, workspace_size_,
  //                                 stream_));
  //   cudaErrCheck(cudaStreamSynchronize(stream_));
  // } else {
  //   caches_.Put(key, BlasLtValueParams{});
  //   auto &params = caches_.Get(key);

  //   params.a_op = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  //   params.b_op = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  //   cublasErrCheck(cublasLtMatmulDescCreate(&params.operation_desc,
  //                                           CUBLAS_COMPUTE_32F, CUDA_R_32F));
  //   cublasErrCheck(cublasLtMatmulDescSetAttribute(
  //       params.operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &params.a_op,
  //       sizeof(params.a_op)));
  //   cublasErrCheck(cublasLtMatmulDescSetAttribute(
  //       params.operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &params.b_op,
  //       sizeof(params.b_op)));

  //   cublasErrCheck(cublasLtMatrixLayoutCreate(
  //       &params.a_desc, CUDA_R_32F, trans_a ? k : m, trans_a ? m : k, lda));
  //   cublasErrCheck(cublasLtMatrixLayoutCreate(
  //       &params.b_desc, CUDA_R_32F, trans_b ? n : k, trans_b ? k : n, ldb));
  //   cublasErrCheck(
  //       cublasLtMatrixLayoutCreate(&params.c_desc, CUDA_R_32F, m, n, ldc));

  //   cublasErrCheck(cublasLtMatmulPreferenceCreate(&params.preference));
  //   cublasErrCheck(cublasLtMatmulPreferenceSetAttribute(
  //       params.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
  //       &workspace_size_, sizeof(workspace_size_)));

  //   const int requested_AlgoCount = 12;
  //   int returned_results = 0;
  //   cublasLtMatmulHeuristicResult_t heuristic_results[requested_AlgoCount] =
  //   {
  //       0};
  //   int bestAlgoIdx = 0;
  //   float time = 0;
  //   float bestAlgoTime = 0;
  //   cudaEvent_t start_event, stop_event;
  //   cudaErrCheck(cudaEventCreate(&start_event));
  //   cudaErrCheck(cudaEventCreate(&stop_event));

  //   cublasErrCheck(cublasLtMatmulAlgoGetHeuristic(
  //       hanlde_, params.operation_desc, params.a_desc, params.b_desc,
  //       params.c_desc, params.c_desc, params.preference, requested_AlgoCount,
  //       heuristic_results, &returned_results));

  //   if (returned_results == 0) {
  //     cublasErrCheck(CUBLAS_STATUS_NOT_SUPPORTED);
  //   }

  //   constexpr int repeatAlgoCheck = 5;
  //   std::vector<float> algoTimes(repeatAlgoCheck);
  //   for (int algoIdx = 0; algoIdx < returned_results; algoIdx++) {
  //     for (int checkIdx = 0; checkIdx < repeatAlgoCheck; checkIdx++) {
  //       cudaErrCheck(cudaEventRecord(start_event, stream_));

  //       cublasErrCheck(cublasLtMatmul(hanlde_, params.operation_desc, &alpha,
  //       A,
  //                                     params.a_desc, B, params.b_desc, &beta,
  //                                     C, params.c_desc, C, params.c_desc,
  //                                     &heuristic_results[algoIdx].algo,
  //                                     workspace_, workspace_size_, stream_));

  //       cudaErrCheck(cudaEventRecord(stop_event, stream_));
  //       cudaErrCheck(cudaEventSynchronize(stop_event));
  //       cudaErrCheck(cudaEventElapsedTime(&time, start_event, stop_event));
  //       algoTimes[checkIdx] = time;
  //     }
  //     time = median(algoTimes);
  //     if (algoIdx == 0 || time < bestAlgoTime) {
  //       bestAlgoTime = time;
  //       bestAlgoIdx = algoIdx;
  //     }
  //   }
  //   params.algo = heuristic_results[bestAlgoIdx].algo;

  //   cublasErrCheck(cublasLtMatmul(hanlde_, params.operation_desc, &alpha, A,
  //                                 params.a_desc, B, params.b_desc, &beta, C,
  //                                 params.c_desc, C, params.c_desc,
  //                                 &params.algo, workspace_, workspace_size_,
  //                                 stream_));
  //   cudaErrCheck(cudaStreamSynchronize(stream_));
  // }
}