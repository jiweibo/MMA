#pragma once

template <typename T>
void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
            T *d_odata);