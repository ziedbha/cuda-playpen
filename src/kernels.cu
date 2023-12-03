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

__global__ void cuda_advc_problem_1_1(uchar* tokens, short* endlines, uint32_t* sum, int totalNumTokens, int totalNumLines)
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

void advc_problem_1_1(uchar* tokens, short* endlines, uint32_t* sum, int totalNumTokens, int totalNumLines, int max_line_size)
{
    assert(64 > max_line_size);
    dim3 blockDims(64, 16); // Each row of threads will handle 1 line

    unsigned int numBlocks = (totalNumLines + blockDims.y - 1) / blockDims.y;
    dim3 gridDims(numBlocks);

    cuda_advc_problem_1_1<<<gridDims, blockDims>>>(tokens, endlines, sum, totalNumTokens, totalNumLines);
    checkCUDAError("Kernel failed!");

    // Device WFI
    checkCUDAError(cudaThreadSynchronize());
}

__constant__ char c_alpha_nums[9][10] = { "one\0", "two\0", "three\0", "four\0", "five\0", "six\0", "seven\0", "eight\0", "nine\0" };

__global__ void cuda_advc_problem_1_2(uchar* tokens, short* endlines, uint32_t* sum, int totalNumTokens, int totalNumLines)
{
    int lineIdx = threadIdx.y + blockDim.y * blockIdx.x;

    __shared__ short digits[64][16]; // digits per line
    __shared__ short2 beginEnd[16]; // bounds of each line held by a block
    __shared__ short numbers[16]; // final numbers, per line, held by a block

    if (lineIdx >= totalNumLines)
    {
        return;
    }

    // Load bounds of a line into shmem
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

    // Load all chars into shmem
    short2 localLineIdx = beginEnd[threadIdx.y];
    if (localLineIdx.x + threadIdx.x <= localLineIdx.y)
    digits[threadIdx.x][threadIdx.y] = tokens[localLineIdx.x + threadIdx.x];
    __syncthreads();

    // All threads in a column drop out except the first
    // TODO Optimization: maybe we can keep a few more around to do the character search?
    if (threadIdx.x != 0)
    {
        return;
    }

    // Loop over chars, and find out if words are converted into numbers
    short maxIdx = -1;
    short minIdx = 100;
    short maxVal = 0;
    short minVal = 0;
    short numChars = localLineIdx.y - localLineIdx.x;
#pragma unroll
    for (int i = 0; i <= numChars; i++)
    {
        short val = digits[i][threadIdx.y];
        bool valid = false;

        if (val >= 48 && val <= 57) // number
        {
            valid = true;
        }
        else // lowercase alpha
        {
            for (int j = 0; j < 9; j++) // iterate over all alpha nums to try to find a match
            {
                char* alphanum = c_alpha_nums[j];
                int k = 0;
                int ii = i;
                bool matched = true;
                while (alphanum[k] != '\0')
                {
                    if (alphanum[k] == digits[ii][threadIdx.y])
                    {
                        k++;
                        ii++;
                    }
                    else
                    {
                        matched = false;
                        break;
                    }
                }

                if (matched)
                {
                    val = j + 1 + 48;
                    valid = true;
                    i = ii - 2; // move forward the base for loop, but not too much because some alpha nums share characters (e.g. eight and two --> eightwo)
                    break;
                }
            }
        }

        if (valid)
        {
            val = val - 48;
            if (maxIdx < i)
            {
                maxIdx = i;
                maxVal = val;
            }

            if (minIdx > i)
            {
                minIdx = i;
                minVal = val;
            }

            if (blockIdx.x == 14 && threadIdx.y == 8) // debug only
            {
                printf("minval %d, maxval %d\n",  minVal, maxVal);
            }
        }
    }

    // just for debug: first block, first line
    if (blockIdx.x == 14 && threadIdx.y == 8)
    {
        printf("begin %d, end %d, range %d\n", beginEnd[threadIdx.y].x, beginEnd[threadIdx.y].y, beginEnd[threadIdx.y].y - beginEnd[threadIdx.y].x + 1);
        for (int i = 0; i < 64; i++) // print this entire line
        {
            printf("val % d\n", digits[i][threadIdx.y]);
        }

        printf("total %d\n", 10 * minVal + maxVal);
    }

    // Store the number of every line in shmem
    numbers[threadIdx.y] = 10 * minVal + maxVal;
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

void advc_problem_1_2(uchar* tokens, short* endlines, uint32_t* sum, int totalNumTokens, int totalNumLines, int max_line_size)
{
    assert(64 > max_line_size);
    dim3 blockDims(64, 16); // Each row of threads will handle 1 line

    unsigned int numBlocks = (totalNumLines + blockDims.y - 1) / blockDims.y;
    dim3 gridDims(numBlocks);

    cuda_advc_problem_1_2 << <gridDims, blockDims >> > (tokens, endlines, sum, totalNumTokens, totalNumLines);
    checkCUDAError("Kernel failed!");

    // Device WFI
    checkCUDAError(cudaThreadSynchronize());
}

__constant__ uchar3 c_bag_limits{ 12, 13, 14 };

__global__ void cuda_advc_problem_2(uchar4* combos, uint32_t* result, int num_combos, int num_games)
{
    if (threadIdx.x > num_combos)
    {
        return;
    }

    extern __shared__ short possible_games[];

    for (int i = threadIdx.x; i < num_games; i += blockDim.x)
        possible_games[i] = 1;
    __syncthreads();

    uchar4 game_data = combos[threadIdx.x];
    if (game_data.x > c_bag_limits.x ||
        game_data.y > c_bag_limits.y ||
        game_data.z > c_bag_limits.z)
    {
        possible_games[game_data.w] = 0;
    }
    __syncthreads();

    if (threadIdx.x >= num_games)
    {
        return;
    }

    if (possible_games[threadIdx.x])
    {
        atomicAdd(&result[0], threadIdx.x + 1);
    }
}

void advc_problem_2(uchar4* combos, uint32_t* result, int num_combos, int num_games)
{
    assert(num_combos <= 1024);
    dim3 blockDims(num_combos); // Each thread will handle 1 game

    dim3 gridDims(1);

    cuda_advc_problem_2 << <gridDims, blockDims, sizeof(short) * num_games >> > (combos, result, num_combos, num_games);
    checkCUDAError("Kernel failed!");

    // Device WFI
    checkCUDAError(cudaThreadSynchronize());
}

