#include <cstdio>

#include "kernels.cuh"

#define __CUDACC__

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <device_functions.h>

void checkCUDAError(cudaError_t err)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s:\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void cuda_advc_problem1(uchar* tokens, short* endlines, uint32_t* sum, int totalNumTokens, int totalNumLines)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    extern __shared__ short numbers[];

    if (idx > totalNumLines)
    {
        return;
    }

    // TODO optimization: reduce global mem access by ~2x by caching the endline of adjacent lines
    short end_line = endlines[idx];
    short begin_line = 0;
    if (idx > 0)
    {
        begin_line = endlines[idx - 1] + 1;
    }

    int num_chars = end_line - begin_line + 1;

    uchar first_digit = 0;
    uchar last_digit = 0;
    bool found_last_digit = false;
#pragma unroll
    for (int i = 0; i < num_chars; i++)
    {
        uchar token = tokens[begin_line + i];
        if (token >= 48 && token <= 57) // ascii range for digits
        {
            if (!found_last_digit)
            {
                last_digit = token - 48;
                found_last_digit = true;
                first_digit = last_digit;
            }
            else
            {
                first_digit = token - 48;
            }
        }
    }

    short number = first_digit + 10 * last_digit;

    numbers[threadIdx.x] = number;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        // loop over shmem to get the total over this entire block
        // TODO optimization: use warp level atomics to reduce shmem size and distribute addition work over the warps
        uint32_t total = 0;
#pragma unroll
        for (int i = 0; i < blockDim.x; i++)
        {
            total += numbers[i];
        }

        // atomically add the accumulated total
        atomicAdd(&sum[0], total);
    }
}

void advc_problem1(uchar* tokens, short* endlines, uint32_t* sum, int totalNumTokens, int totalNumLines)
{
    // set up crucial magic
    unsigned int blockSize = 256; // Each block will handle 1 line
    dim3 blockDims(blockSize);

    unsigned int numBlocks = (totalNumLines + blockSize - 1) / blockSize;
    dim3 gridDims(numBlocks);

    int shmemSize = blockSize * sizeof(short);
    checkCUDAError(cudaFuncSetAttribute(cuda_advc_problem1, cudaFuncAttributeMaxDynamicSharedMemorySize, shmemSize));

    cuda_advc_problem1<<<gridDims, blockDims, shmemSize >>>(tokens, endlines, sum, totalNumTokens, totalNumLines);
    checkCUDAError("Kernel failed!");

    // Device WFI
    checkCUDAError(cudaThreadSynchronize());
}

