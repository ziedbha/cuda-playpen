#include <cstdio>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <assert.h>

// CUDA
#include <cuda_runtime_api.h>

// externals
#include <argparse.hpp>

#include "main.hpp"

void runProblem1(std::string inputPath)
{
    // Read problem data as:
    // tokens:      ascii counterpart of every character, excluding newlines
    // endline_idx: index of the last character of everyline. The indices are w.r.t the tokens array

    // The way the problem is solved is by doing a reduction, sigh
    std::ifstream input_file(inputPath);
    std::vector<unsigned char> tokens;
    tokens.reserve(10000);
    std::vector<short> endline_idx;
    if (input_file.is_open())
    {
        std::cout << "Input data:" << std::endl;
        size_t current_size = 0;
        while (input_file.good())
        {
            std::string line;
            input_file >> line;

            for (auto& c : line)
            {
                tokens.push_back((unsigned char)c);
            }
            endline_idx.push_back((short)(tokens.size() - 1));
        }
    }

    // Allocate data on device and copy it over
    void* cuTokens = nullptr;
    void* cuIndices = nullptr;
    void* cuSum = nullptr;
    checkCUDAError(cudaMalloc(&cuTokens, sizeof(uchar) * tokens.size()));       // in
    checkCUDAError(cudaMalloc(&cuIndices, sizeof(short) * endline_idx.size())); // in
    checkCUDAError(cudaMalloc(&cuSum, sizeof(uint32_t)));                       // out, single num

    checkCUDAError(cudaMemcpy(cuTokens, tokens.data(), tokens.size() * sizeof(uchar1), cudaMemcpyHostToDevice));
    checkCUDAError(cudaMemcpy(cuIndices, endline_idx.data(), endline_idx.size() * sizeof(short1), cudaMemcpyHostToDevice));

    // Run problem on device
    // TODO optimization: right now, the data is setup such that each thread will handle 1 line
    // this means each thread will do ~line length gmem accesses
    // Within a warp, the work is not evenly distributed: some lines are longer than others. We should try to normalize line lengths somehow...
    advc_problem1((uchar*)cuTokens, (short*)cuIndices, (uint32_t*)cuSum, tokens.size(), endline_idx.size());

    // Copy over data from device to host
    uint32_t sum = 0;
    checkCUDAError(cudaMemcpy(&sum, cuSum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Sum = " << sum << ", correct sum = " << 55108 << std::endl;
    assert(sum == 55108);
}

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("AdventOfCode2023");

    program.add_argument("--problem", "-p")
        .help("Runs a given Advent Of Code 2023 problem")
        .required()
        .default_value(1)
        .scan<'i', int>();

    program.add_argument("--path")
        .help("Path to a given Advent Of Code 2023 problem")
        .required()
        .default_value("");

    // Parse args
    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Verify args are OK
    auto in_path = program.get<std::string>("--path");
    if (!std::filesystem::exists(in_path))
    {
        std::cout << "ERROR: path " << in_path << " does not exist" << std::endl;
        exit(1);
    }

    // Initialize CUDA and run problem
    if (!initCUDA())
    {
        std::cout << "ERROR: CUDA could not be initialized" << std::endl;
        exit(1);
    }
    else
    {
        if (program.get<int>("--problem") == 1)
        {
            runProblem1(in_path);
        }
    }

    return 0;
}

bool initCUDA()
{
    std::string deviceName;
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout << "ERROR: GPU device number is greater than the number of devices!" <<
                  "Perhaps a CUDA-capable GPU is not installed?" << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    auto major = deviceProp.major;
    auto minor = deviceProp.minor;

    std::ostringstream ss;
    std::cout << "[SM " << major << "." << minor << "] " << deviceProp.name << std::endl;

    return true;
}