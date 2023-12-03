#include <cstdio>

#include "kernels.cuh"
#include <assert.h>

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
    int lineIdx = threadIdx.y + blockDim.y * blockIdx.x;

    __shared__ short digits[64][16]; // digits per line
    __shared__ short2 beginEnd[16]; // bounds of each line held by a block
    __shared__ short numbers[16]; // final numbers, per line, held by a block
    __shared__ short2 maxMinPerLine[16][2];

    if (lineIdx >= totalNumLines)
    {
        return;
    }

    if (threadIdx.x == 0) // first thread of every column will load the line data
    {
        beginEnd[threadIdx.y].y = endlines[lineIdx];
        if (lineIdx > 0)
        {
            beginEnd[threadIdx.y].x = endlines[lineIdx - 1] + 1;
        }
        else
        {
            beginEnd[threadIdx.y].x = 0;
        }
    }
    __syncthreads();

    short val = 0;
    short maxIdx = -1;
    short minIdx = 100;
    short temp = 0;
    short2 localLineIdx = beginEnd[threadIdx.y];
    short numChars = localLineIdx.y - localLineIdx.x;

    val = tokens[localLineIdx.x + threadIdx.x] - 48;
    if (val >= 0 && val <= 9 && threadIdx.x <= numChars)
    {
        digits[threadIdx.x][threadIdx.y] = val;
        maxIdx = threadIdx.x;
        minIdx = threadIdx.x;
    }

    // warp-wide max/min
    __syncwarp();
    for (int i = 16; i > 0; i /= 2) // warp synced max/min
    {
        temp = __shfl_down_sync(0xffffffff, maxIdx, i);
        maxIdx = max(maxIdx, temp);

        temp = __shfl_down_sync(0xffffffff, minIdx, i);
        minIdx = min(minIdx, temp);
    }

    // Write max and min to shmem
    if (threadIdx.x == 0) // warp #1
    {
        maxMinPerLine[threadIdx.y][0] = make_short2(minIdx, maxIdx);
    }
    else if (threadIdx.x == 32) // warp #2
    {
        maxMinPerLine[threadIdx.y][1] = make_short2(minIdx, maxIdx);
    }
    __syncthreads();

    // only first column of threads is needed since all per-line data is gathered
    if (threadIdx.x != 0) 
    {
        return;
    }

    // Write the full number
    short minn = min(maxMinPerLine[threadIdx.y][0].x, maxMinPerLine[threadIdx.y][1].x);
    short maxx = max(maxMinPerLine[threadIdx.y][0].y, maxMinPerLine[threadIdx.y][1].y);

    // just for debug: first block, first line
    if (blockIdx.x == 0 && threadIdx.y == 0) 
    {
        printf("begin %d, end %d, range %d\n", beginEnd[threadIdx.y].x, beginEnd[threadIdx.y].y, beginEnd[threadIdx.y].y - beginEnd[threadIdx.y].x + 1);
        printf("warp #1 min idx %d, max idx %d\n", maxMinPerLine[threadIdx.y][0].x, maxMinPerLine[threadIdx.y][0].y );
        printf("warp #2 min idx %d, max idx %d\n", maxMinPerLine[threadIdx.y][1].x, maxMinPerLine[threadIdx.y][1].y);
        for (int i = 0; i < 64; i++) // print this entire line
        {
            printf("val % d\n",  digits[i][threadIdx.y]);
        }

        printf("total %d\n", 10 * digits[minn][threadIdx.y] + digits[maxx][threadIdx.y]);
    }       

    // Store the number of every line in shmem
    numbers[threadIdx.y] = 10 * digits[minn][threadIdx.y] + digits[maxx][threadIdx.y];
    __syncthreads();

    // Only first thread per block is needed t odo the reduction
    if (threadIdx.y == 0)
    {
        // loop over shmem to get the total over this entire block
        uint32_t total = 0;
#pragma unroll
        for (int i = 0; i < blockDim.y; i++)
        {
            total += numbers[i];
        }

        // atomically add the accumulated total
        atomicAdd(&sum[0], total);
    }
}

void advc_problem1(uchar* tokens, short* endlines, uint32_t* sum, int totalNumTokens, int totalNumLines, int max_line_size)
{
    assert(64 > max_line_size);
    dim3 blockDims(64, 16); // Each row of threads will handle 1 line

    unsigned int numBlocks = (totalNumLines + blockDims.y - 1) / blockDims.y;
    dim3 gridDims(numBlocks);

    //int shmemSize = blockSize * sizeof(short);
    //checkCUDAError(cudaFuncSetAttribute(cuda_advc_problem1, cudaFuncAttributeMaxDynamicSharedMemorySize, shmemSize));

    cuda_advc_problem1<<<gridDims, blockDims>>>(tokens, endlines, sum, totalNumTokens, totalNumLines);
    checkCUDAError("Kernel failed!");

    // Device WFI
    checkCUDAError(cudaThreadSynchronize());
}

