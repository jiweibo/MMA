#include "ops/binary/binary.h"

// #include <cutlass/cutlass.h>
// #include <cutlass/half.h>

#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor.hpp"

using namespace cute;

// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add_local_tile_multi_elem_per_thread_half(half *z, int num, const half *x, const half *y,
                                                                 const half a, const half b, const half c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / kNumElemPerThread) {
    // TODO:...
    return;
  }
  Tensor tz = cute::make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = cute::make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = cute::make_tensor(make_gmem_ptr(y), make_shape(num));

  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  copy(txr, txR);
  copy(tyr, tyR);

  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  auto txR2 = recast<half2>(txR);
  auto tyR2 = recast<half2>(tyR);
  auto tzR2 = recast<half2>(tzR);

#pragma unroll
  for (int i = 0; i < size(tzR2); ++i) {
    // two hfma instruction
    tzR2(i) = a2 * txR2(i) + (b2 * tyR2(i) + c2);
  }

  auto tzRx = recast<half>(tzR2);

  // STG.128
  copy(tzRx, tzr);
}

template <>
void axbyc<half>(half *z, const half *x, const half *y, half a, half b, half c, int num, cudaStream_t stream) {
  dim3 block = 256;
  dim3 grid = (num + block.x - 1) / block.x;

  vector_add_local_tile_multi_elem_per_thread_half<2><<<grid, block, 0, stream>>>(z, num, x, y, a, b, c);
}