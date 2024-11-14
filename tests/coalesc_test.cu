#include "cutlass/layout/vector.h"
#include <cuda_runtime_api.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>

template <typename T>
__global__ void OffsetCopy(T *odata, T *idata, int num, int offset) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
  if (xid < num) {
    odata[xid] = idata[xid];
  }
}


struct __align__(32) float6 {
  float x;
  float y;
};
using TESTTYPE = float6;
// using TESTTYPE = float4;

void TestCopy() {
  LOG(INFO) << "sizeof " << sizeof(float6);
  cutlass::HostTensor<TESTTYPE, cutlass::layout::ColumnMajor> x({1024, 1024});
  cutlass::HostTensor<TESTTYPE, cutlass::layout::ColumnMajor> z({1024, 1024});

  cutlass::reference::host::TensorFill(x.host_view(), TESTTYPE{1.0f});
  cutlass::reference::host::TensorFill(z.host_view(), TESTTYPE{0.f});

  x.sync_device();
  z.sync_device();

  auto *x_data = x.device_data();
  auto *z_data = z.device_data();

  dim3 block = 256;
  dim3 grid = (x.size() + block.x - 1) / block.x;
  OffsetCopy<<<grid, block, 0>>>(z_data, x_data, x.size(), 0);
}

int main() {
  TestCopy();
  return 0;
}