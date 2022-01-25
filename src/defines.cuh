#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <fstream>
#include <stdio.h>
#include <random>
#include <H5Cpp.h>
#include <chrono>
#include <vector>

// we need these includes for CUDA's random number stuff
#include <curand.h>
#include <curand_kernel.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//! Just some utility functions that can be used throughout the codebase.
namespace Utility
{
    std::default_random_engine& UnseededRandomEngine();
	std::default_random_engine& SeededRandomEngine();
};

#define USE_DOUBLE_PRECISION 0 
#if USE_DOUBLE_PRECISION
typedef double Float;
#else
typedef float Float;
#endif // USE_DOUBLE_PRECISION

// TODO : SET NUMBER OF THREADS DYNAMICALLY BASED ON CARD USED
#define NTHREADS 512 
#define NBLOCKS 32
#define NCHUNKS 10
#define N_PER_CHUNK 10000
