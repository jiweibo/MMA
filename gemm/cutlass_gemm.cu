#include "cute/algorithm/axpby.hpp"
#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/arch/copy.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/config.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include "cute/swizzle_layout.hpp"
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

#include "cute/util/debug.hpp"
#include "cute/util/print.hpp"
#include "cutlass/uint128.h"
#include "gemm/cutlass_gemm.h"

template <class ProblemShape, class CtaTiler, class TA, class AStride, class ASmemLayout,
          class AThreadLayout, class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void gemm_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const *A, AStride dA, ASmemLayout sA_layout,
    AThreadLayout tA, TB const *B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB, TC *C,
    CStride dC, CSmemLayout, CThreadLayout tC, Alpha alpha, Beta beta) {
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB)); // NumThreads
  CUTE_STATIC_ASSERT_V(size(tC) == size(tA)); // NumThreads

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{}); // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{}); // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{}); // BLK_N / THR_N
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{}); // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{}); // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{}); // BLK_N / THR_N

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MK

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M, N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m, n, k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //
  // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB over
  // data A|B tiles
  Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M, THR_K, k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M, THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N, THR_K, k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N, THR_K)

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA)); // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // THR_K
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB)); // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // THR_K

  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via projections of a ThreadLayout tC

  // Partition sA (M,K) by the rows of tC
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M, BLK_K)
  // Partition sB (N,K) by the cols of tC
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N, BLK_K)
  // Partition gC (M,N) by the tile of tC
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M, THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  Tensor tCrC = make_tensor_like(tCgC); // (THR_M, THR_N)

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC)); // THR_M
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA)); // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC)); // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB)); // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB)); // BLK_K

  // Clear the accumulators
  clear(tCrC);

#if 0
  if (thread0()) {
    print("  mA:  ");
    print(mA);
    print("\n");
    print("  gA:  ");
    print(gA);
    print("\n");
    print("  sA:  ");
    print(sA);
    print("\n");
    print("tAgA:  ");
    print(tAgA);
    print("\n");
    print("tAsA:  ");
    print(tAsA);
    print("\n");
  }
#endif

#if 0
  if (thread0()) {
    print("  mB:  ");
    print(mB);
    print("\n");
    print("  gB:  ");
    print(gB);
    print("\n");
    print("  sB:  ");
    print(sB);
    print("\n");
    print("tBgB:  ");
    print(tBgB);
    print("\n");
    print("tBsB:  ");
    print(tBsB);
    print("\n");
  }
#endif

#if 0
  if (thread(0)) {
    print("  mC:  ");
    print(mC);
    print("\n");
    print("  gC:  ");
    print(gC);
    print("\n");
    print("tCsA:  ");
    print(tCsA);
    print("\n");
    print("tCsB:  ");
    print(tCsB);
    print("\n");
    print("tCgC:  ");
    print(tCgC);
    print("\n");
    print("tCrC:  ");
    print(tCrC);
    print("\n");

    // print_latex(tCsA.layout());
    // print_latex(tCsB.layout());
    // print_latex(tCgC.layout());
  }
#endif

  // TUTORIAL: Example of a simple mainloop that read tiles of data into shared
  // memory,
  //           and then computes on those tiles.
  //   copy(.) operates on the global and shared memory via the tA|tB
  //   partitioning gemm(.) operates on the shared and register memory via the
  //   tC partitioning

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
// Copy gmem to smem with tA|tB thread-partitioned tensors
#if 0
    if(thread0()) {
      print_latex(tAgA(_, _, k_tile).layout());
      print("\n");
      print_latex(tAsA.layout());
      print("\n");
      return;
    }
#endif
    copy(tAgA(_, _, k_tile), tAsA); // A (THR_M, THR_K) -> (THR_M, THR_K)
    copy(tBgB(_, _, k_tile), tBsB); // B (THR_N, THR_K) -> (THR_N, THR_K)

    // TUTORIAL: The above call to copy(tAgA(_,_,k_tile), tAsA) is equivalent to
    //   Tensor tAgAk = tAgA(_, _, k_tile);
    //   CUTE_UNROLL
    //   for (int i = 0; i < size(tAsA); ++i) {
    //     tAsA(i) = tAgAk(i);
    //   }

    cp_async_fence();   // Label the end of (potential) cp.async instructions
    cp_async_wait<0>(); // Sync on all (potential) cp.async instructions
    __syncthreads();    // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC); // (THR_M, THR_N) += (THR_M, BLK_K) * (THR_N, BLK_K)

    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         tCrC(m, n) += tCsA(m, k) * tCsB(n, k);
    //       }
    //     }
    //   }

    __syncthreads(); // Wait for all threads to read from smem
  }

  //
  // Epilogue
  //
  axpby(alpha, tCrC, beta, tCgC);

  // TUTORIAL: The above call to axpby(alpha, tCrC, beta, tCgC) is equivalent to
  CUTE_UNROLL
  for (int i = 0; i < size(tCsA); ++i) {
    tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  }
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
             Beta beta, TC *C, int ldC, cudaStream_t stream) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK)); // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK)); // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN)); // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{})); // (m,n) -> thr_idx

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB,
                                                C, dC, sC, tC, alpha, beta);
}

// Setup params for a TN GEMM
// Use padded m-major smem sA, padded n-major smem sB, and k-major threads tA|tB
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
             Beta beta, TC *C, int ldC, cudaStream_t stream) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M,N,K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK), LayoutRight{}); // (m,k) -> smem_idx; k-major
  auto sB = make_layout(make_shape(bN, bK), LayoutRight{}); // (n,k) -> smem_idx; k-major
  auto sC = make_layout(make_shape(bM, bN));                // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA =
      make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}); // (m,k) -> thr_idx; k-major
  auto tB =
      make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}); // (n,k) -> thr_idx; k-major
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));         // (m,n) -> thr_idx; m-major

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB,
                                                C, dC, sC, tC, alpha, beta);
}

/// C = alpha * A * B + beta * C
/// layout(A) is (M, K) : (1, lda), m-major
/// layout(B) is (N, K) : (1, ldb), n-major
/// layout(C) is (M, N) : (1, ldc), m-major
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_mnm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream) {
  gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

/// C = alpha * A * B + beta * C
/// layout(A) is (M, K) : (1, lda), m-major
/// layout(B) is (N, K) : (ldb, 1), k-major
/// layout(C) is (M, N) : (1, ldc), m-major
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_mkm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M,N,K)

  // Define TN strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK), LayoutRight{}); // (n,k) -> smem_idx; k-major
  auto sC = make_layout(make_shape(bM, bN));                // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{})); // (m,k) -> thr_idx; m-major
  auto tB =
      make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}); // (n,k) -> thr_idx; k-major
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));         // (m,n) -> thr_idx; m-major

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB,
                                                C, dC, sC, tC, alpha, beta);
}

/// C = alpha * A * B + beta * C
/// layout(A) is (M, K) : (lda, 1), k-major
/// layout(B) is (N, K) : (1, ldb), n-major
/// layout(C) is (M, N) : (1, ldc), m-major
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_knm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M,N,K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK), LayoutRight{}); // (m,k) -> smem_idx; k-major
  auto sB = make_layout(make_shape(bN, bK));                // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA =
      make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}); // (m,k) -> thr_idx; k-major
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));          // (n,k) -> thr_idx; n - major
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));         // (m,n) -> thr_idx; m-major

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB,
                                                C, dC, sC, tC, alpha, beta);
}

/// C = alpha * A * B + beta * C
/// layout(A) is (M, K) : (lda, 1), k-major
/// layout(B) is (N, K) : (ldb, 1), k-major
/// layout(C) is (M, N) : (1, ldc), m-major
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_kkm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream) {
  gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}

template void gemm_tn<float, float, float, float, float>(int m, int n, int k, float alpha,
                                                         float const *A, int ldA, float const *B,
                                                         int ldB, float beta, float *C, int ldC,
                                                         cudaStream_t stream);

template void gemm_nt<float, float, float, float, float>(int m, int n, int k, float alpha,
                                                         float const *A, int ldA, float const *B,
                                                         int ldB, float beta, float *C, int ldC,
                                                         cudaStream_t stream);

template void gemm_mnm<float, float, float, float, float>(int m, int n, int k, float alpha,
                                                          float const *A, int ldA, float const *B,
                                                          int ldB, float beta, float *C, int ldC,
                                                          cudaStream_t stream);

template void gemm_mkm<float, float, float, float, float>(int m, int n, int k, float alpha,
                                                          float const *A, int ldA, float const *B,
                                                          int ldB, float beta, float *C, int ldC,
                                                          cudaStream_t stream);

template void gemm_knm<float, float, float, float, float>(int m, int n, int k, float alpha,
                                                          float const *A, int ldA, float const *B,
                                                          int ldB, float beta, float *C, int ldC,
                                                          cudaStream_t stream);

template void gemm_kkm<float, float, float, float, float>(int m, int n, int k, float alpha,
                                                          float const *A, int ldA, float const *B,
                                                          int ldB, float beta, float *C, int ldC,
                                                          cudaStream_t stream);

// ---------------------------------------------------------------------------------

template <class ProblemShape, class CtaTiler, class TA, class AStride, class ASmemLayout,
          class TiledCopyA, class TB, class BStride, class BSmemLayout, class TiledCopyB, class TC,
          class CStride, class CSmemLayout, class TiledMma, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device2(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const *A, AStride dA, ASmemLayout sA_layout,
    TiledCopyA copy_a, TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, TC *C,
    CStride dC, CSmemLayout, TiledMma mma, Alpha alpha, Beta beta) {
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M, N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m, n, k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N, BLK_N, k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of partitioning via a TiledCopy

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tBrB = make_fragment_like(tBsB); // (CPY, CPY_N, CPY_K)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA)); // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA)); // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB)); // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB)); // CPY_K

  // Copy gmem to rmem for k_tile=0
  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);

  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via a TiledMMA
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA); // (MMA, MMA_M, MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB); // (MMA, MMA_N, MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

  // Allocate the acumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N)

  CUTE_STATIC_ASSERT_V(shape(tCrC) == shape(tCgC));     // (MMA, MMA_M, MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA)); // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB)); // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB)); // MMA_K

  // Clear the accumulators
  clear(tCrC);

#if 0
  if (thread0()) {
    print("mA: ");
    print(mA);
    print("\n");
    print("gA: ");
    print(gA);
    print("\n");
    print("sA: ");
    print(sA);
    print("\n");
    print("tAgA: ");
    print(tAgA);
    print("\n");
    print("tAsA: ");
    print(tAsA);
    print("\n");
    print("tArA: ");
    print(tArA);
    print("\n");
  }

  if (thread0()) {
    print("mB: ");
    print(mB);
    print("\n");
    print("gB: ");
    print(gB);
    print("\n");
    print("sB: ");
    print(sB);
    print("\n");
    print("tBgB: ");
    print(tBgB);
    print("\n");
    print("tBsB: ");
    print(tBsB);
    print("\n");
  }

  if (thread0()) {
    print("mC: ");
    print(mC);
    print("\n");
    print("gC: ");
    print(gC);
    print("\n");
    print("tCsA: ");
    print(tCsA);
    print("\n");
    print("tCsB: ");
    print(tCsB);
    print("\n");
    print("tCgC: ");
    print(tCgC);
    print("\n");
    print("tCrC: ");
    print(tCrC);
    print("\n");
  }
#endif

  // TUTORIAL: Example of an inner loop that pipelines compute with reads
  //           from global memory by staging through register and shared memory.
  //   Data is read from global to registers, then to shared via the TieldCopy partitions
  //   gemm(.) operates on the shared memory directly via the TiledMMA partitions

  auto K_TILE_MAX = size<3>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    // Copy rmem to smem with tA|tB thread-partitioned tensors
    __syncthreads(); // Wait for all threads to consume smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads(); // Wait for all threads to consume smem

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);
    // TUTORIAL: The above call to copy(copy_a, ...) is equivalent to
    // CUTE_UNROLL
    // for (int k = 0; k < size<1>(tCsA); ++k) {
    //   CUTE_UNROLL
    //   for (int m = 0; m < size<0>(tCrC); ++m) {
    //     copy_a.call(tAgA(_, m, k), tArA(_, m, k));
    //   }
    // }

    // Compute gemm on mma-partitioned smem
    gemm(mma, tCsA, tCsB, tCrC);
    // TUTORIAL: The above call to gemm(tCsA, ...) is equivalent to
    // CUTE_UNROLL
    // for (int k = 0; k < size<1>(tCsA); ++k) {
    //   CUTE_UNROLL
    //   for (int m = 0; m < size<0>(tCrC); ++m) {
    //     CUTE_UNROLL
    //     for (int n = 0; n < size<1>(tCrC); ++n) {
    //       mma.call(tCsA(_, m, k), tCsB(_, n, k), tCrC(_, m, n));
    //     }
    //   }
    // }
  }

  //
  // Epilogue
  //
  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt2(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK)); // (m, k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK)); // (n, k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN)); // (m, n) -> smem_idx; m-major

  // Define the thread layouts (static)

  // TUTORIAL: Construct TiledCopy with a particular Copy_Atom to use and
  //           define the partitioning pattern to apply.
  // Each thread will (try to) copy 4x1 elements of type TA using 128-bit copy.
  // Use 32x8 of these threads.

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                    Layout<Shape<_32, _8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape<_4, _1>>{}); // Val layout 4x1 major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                                    Layout<Shape<_32, _8>>{}, // Thr layout 32x8 n-major
                                    Layout<Shape<_4, _1>>{}); // Val layout 4x1 n-major

  // TUTORIAL: Construct TiledMMA with particular MMA_Atom to use and
  //           define the partitioning pattern to apply.
  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC, TA, TB>{},
                                 Layout<Shape<_16, _16, _1>>{}); // 16x16x1 UniversalFMA

#if 0
  print("copyA\n");
  print(copyA);
  print("copyB\n");
  print(copyB);
  print("\n");
  print("mmaC\n");
  print(mmaC);
  print("\n");

  print_latex(copyA);
  print("\n");
  print_latex(copyB);
  print("\n");
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device2<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, copyA, B, dB, sB,
                                                 copyB, C, dC, sC, mmaC, alpha, beta);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn2(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B, int ldB,
              Beta beta, TC *C, int ldC, cudaStream_t stream) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  // TODO(wilber)
  auto sA = make_layout(make_shape(bM, bK),
                        make_stride(Int<1>{}, bM)); // (m, k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape(bN, bK),
                        make_stride(Int<1>{}, bN)); // (n, k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bM, bN)); // (m, n) -> smem_idx

  // TUTORIAL: Construct TiledCopy to define the Copy_Atom to use and the
  //           partitioning pattern to apply.
  // Each thread will copy 1x1 elements of type TA.
  // Use 32x8 of these threads arranged in k-major

  // TiledCopy copyA =
  //     make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
  //                     Layout<Shape<_32, _8>, Stride<_8, _1>>{}, // Thr layout 32x8 k-major
  //                     Layout<Shape<_1, _1>>{});                 // Var layout 1x1
  // TiledCopy copyB =
  //     make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
  //                     Layout<Shape<_32, _8>, Stride<_8, _1>>{}, // Thr layout 32x8 k-major
  //                     Layout<Shape<_1, _1>>{});                 // Var layout 1x1

  TiledCopy copyA =
      make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                      Layout<Shape<_32, _8>, Stride<_1, _32>>{}, // Thr layout 32x8 k-major
                      Layout<Shape<_1, _1>>{});                 // Var layout 1x1
  TiledCopy copyB =
      make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
                      Layout<Shape<_32, _8>, Stride<_1, _32>>{}, // Thr layout 32x8 k-major
                      Layout<Shape<_1, _1>>{});                 // Var layout 1x1

  // TUTORIAL: Construct TiledMMA to define the MMA_Atom to use and the
  //           partitioning pattern to apply.
  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.
  TiledMMA mmaC =
      make_tiled_mma(UniversalFMA<TC, TA, TB>{}, Layout<Shape<_16, _16, _1>>{}); // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device2<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, copyA, B, dB, sB,
                                                 copyB, C, dC, sC, mmaC, alpha, beta);
}

template void gemm_nt2<float, float, float, float, float>(int m, int n, int k, float alpha,
                                                          float const *A, int ldA, float const *B,
                                                          int ldB, float beta, float *C, int ldC,
                                                          cudaStream_t stream);

template void gemm_tn2<float, float, float, float, float>(int m, int n, int k, float alpha,
                                                          float const *A, int ldA, float const *B,
                                                          int ldB, float beta, float *C, int ldC,
                                                          cudaStream_t stream);