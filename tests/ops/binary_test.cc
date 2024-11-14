#include "ops/binary/binary.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cutlass/layout/layout.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>


TEST(Binary, axbyc) {
  cutlass::HostTensor<half, cutlass::layout::ColumnMajor> x({1024, 1024});
  cutlass::HostTensor<half, cutlass::layout::ColumnMajor> y({1024, 1024});
  cutlass::HostTensor<half, cutlass::layout::ColumnMajor> z({1024, 1024});

  cutlass::reference::host::TensorFill(x.host_view(), half{1.f});
  cutlass::reference::host::TensorFill(y.host_view(), half{1.f});
  cutlass::reference::host::TensorFill(z.host_view(), half{0.f});

  x.sync_device();
  y.sync_device();
  z.sync_device();

  auto *x_data = x.device_data();
  auto *y_data = y.device_data();
  auto *z_data = z.device_data();

  half a{1.f};
  half b{1.f};
  half c{0.f};

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  axbyc(z_data, x_data, y_data, a, b, c, x.size(), stream);
  cudaStreamSynchronize(stream);

  z.sync_host();
  EXPECT_EQ(z.host_ref().at({0, 0}), half{2.f});
}