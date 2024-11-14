#pragma once

#include <cuda_runtime_api.h>

// z = a*x + b*y + c
template <typename T>
void axbyc(T *z, const T *x, const T *y, T a, T b, T c, int num, cudaStream_t stream);