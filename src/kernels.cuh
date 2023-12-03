#pragma once

#include <cuda.h>
#include <vector_types.h>

typedef unsigned char uchar;

void checkCUDAError(cudaError_t err);
void checkCUDAError(const char* msg);

void advc_problem1(uchar* tokens, short* endlines, uint32_t* sum, int totalNumTokens, int totalNumLines, int max_line_size);
