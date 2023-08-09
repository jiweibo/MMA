
#include "reduction.h"
#include "common.h"
#include "timer.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>

//#include <getopt.h>
#include <iostream>
//#include <unistd.h>

//static const struct option long_opts[] = {
//    {"number", required_argument, NULL, 'n'},
//    {"which", required_argument, NULL, 'w'},
//    {"help", no_argument, NULL, 'h'},
//    {NULL, 0, NULL, 0}};

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
  cudaDeviceProp prop;
  int device;
  cudaErrCheck(cudaGetDevice(&device));
  cudaErrCheck(cudaGetDeviceProperties(&prop, device));

  if (whichKernel < 3) {
    threads = n < maxThreads ? nextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
  } else {
    threads = (n < maxThreads * 2) ? nextPow2((n+1) / 2) : maxThreads;
    blocks = (n + threads * 2 - 1) / (threads * 2);
  }

  if (whichKernel == 6) {
      blocks = maxBlocks < blocks ? maxBlocks : blocks;
  }
}

template <typename T>
T benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                  int maxBlocks, int whichKernel, int testIterations,
                  bool cpuFinalReduction, int cpuFinalThreshold,
                  StopWatchTimer &timer, T *h_odata, T *d_idata, T *d_odata) {
  T gpu_result = 0;
  bool needReadBack = true;

  T *d_intermediateSums;
  cudaErrCheck(cudaMalloc((void **)&d_intermediateSums, sizeof(T) * numBlocks));

  for (int i = 0; i < testIterations; ++i) {
    gpu_result = 0;

    cudaDeviceSynchronize();
    timer.Start();

    // execute the kernel
    reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);
    getLastCudaError("Kernel execution failed");

    if (cpuFinalReduction) {
      cudaErrCheck(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(T),
                              cudaMemcpyDeviceToHost));
      for (int i = 0; i < numBlocks; ++i) {
        gpu_result += h_odata[i];
      }

      needReadBack = false;
    }
    else {
        // sum partial block sums on GPU
        int s = numBlocks;
        int kernel = whichKernel;

        while (s > cpuFinalThreshold) {
            int threads = 0, blocks = 0;
            getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
            cudaErrCheck(cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(T), cudaMemcpyDeviceToDevice));
            reduce<T>(s, threads, blocks, kernel, d_intermediateSums, d_odata);

            if (kernel < 3) {
                s = (s + threads - 1) / threads;
            }
            else {
                s = (s + (threads * 2 - 1)) / (threads * 2);
            }
        }

        if (s > 1) {
            // copy result from device to host
            cudaErrCheck(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));
            for (int i = 0; i < s; ++i) {
                gpu_result += h_odata[i];
            }
            needReadBack = false;
        }
    }

    cudaDeviceSynchronize();
    timer.Stop();
  }

  if (needReadBack) {
    cudaErrCheck(
        cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
  }
  cudaErrCheck(cudaFree(d_intermediateSums));
  return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template <typename T>
T reduceCPU(T* data, int size) {
  T sum = data[0];
  T c = (T)0;

  for (int i = 1; i < size; ++i) {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

template <typename T> bool runTest(int number, int whichKernel) {
  int size = number;
  int maxThreads = 512;
  bool cpuFinalReduce = true;

  T *h_idata = (T *)malloc(size * sizeof(T));
  for (int i = 0; i < size; ++i) {
    h_idata[i] = (rand() & 0xFF) / (T)RAND_MAX;
  }

  int numBlocks = 0;
  int numThreads = 0;

  int maxBlocks = 65535;
  getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
                         numThreads);
  std::cout << "numBlocks = " << numBlocks << ", numThreads = " << numThreads << std::endl;

  T *h_odata = (T *)malloc(numBlocks * sizeof(T));

  T *d_idata = NULL;
  T *d_odata = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_idata, size * sizeof(T)));
  cudaErrCheck(cudaMalloc((void **)&d_odata, numBlocks * sizeof(T)));

  cudaErrCheck(
      cudaMemcpy(d_idata, h_idata, size * sizeof(T), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(T),
                          cudaMemcpyHostToDevice));

  // warmup
  reduce<T>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);

  int testIterations = 100;
  StopWatchTimer timer;

  bool cpuFinalReduction = false;
  int cpuFinalThreshold = 1024;

  T gpu_result =
      benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
                      whichKernel, testIterations, cpuFinalReduction,
                      cpuFinalThreshold, timer, h_odata, d_idata, d_odata);
  double reduceTime = timer.GetAverageTime();
  std::cout << "Reduction, Bandwidth = "
            << size * sizeof(T) * 1e-9 * 1. / (reduceTime * 1e-3)
            << " GB/s, Time = " << reduceTime << " ms, Size = " << size
            << " Elements" << std::endl;

  // compute reference solution
  T cpu_result = reduceCPU<T>(h_idata, size);
  double threshold = 1e-8 * size;
  double diff = 0;
  diff = fabs(gpu_result - cpu_result);
  std::cout << "GPU result = " << gpu_result << std::endl;
  std::cout << "CPU result = " << cpu_result << std::endl;

  cudaErrCheck(cudaFree(d_idata));
  cudaErrCheck(cudaFree(d_odata));
  free(h_idata);
  free(h_odata);

  return diff < threshold;
}

int main(int argc, char **argv) {
  std::cout << argv[0] << " Starting...\n" << std::endl;

  int number = 1 << 25;
  int whichKernel = 0;
  //while (1) {
  //  int option_index = 0;
  //  int c = getopt_long(argc, argv, "n:w:h", long_opts, &option_index);
  //  if (c == -1)
  //    break;
  //  switch (c) {
  //  case 'n':
  //    number = atoi(optarg);
  //    break;
  //  case 'w':
  //    whichKernel = atoi(optarg);
  //    break;
  //  case 'h':
  //    break;
  //  default:
  //    abort();
  //  }
  //}

  whichKernel = atoi(argv[1]);

  std::cout << "number: " << number << std::endl;
  std::cout << "which_kernel: " << whichKernel << std::endl;

  cudaDeviceProp deviceProp;
  int dev;
  int devID = 0;
  cudaErrCheck(cudaSetDevice(devID));
  int major = 0, minor = 0;
  cudaErrCheck(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  cudaErrCheck(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  printf("GPU Device %d: with compute capability %d.%d\n\n", devID, major,
         minor);

  runTest<float>(number, whichKernel);

  return 0;
}