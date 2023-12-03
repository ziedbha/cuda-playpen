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

void runProblem_2(std::string inputPath)
{
    // Input data will be a bunch of uchar4s packed as such: r = red, g = green, b = blue, a = game ID
    std::ifstream input_file(inputPath);
    std::vector<uchar4> combos;
    combos.reserve(10000);
    int num_games = 0;

    if (input_file.is_open())
    {
        size_t current_size = 0;
        std::string line;
        while (std::getline(input_file, line))
        {
            if (line == "")
            {
                continue;
            }

            uchar4 game_vals{}; // r g b game_id

            // "Game x"
            std::string delimiter = ":";
            size_t pos = line.find(delimiter);
            std::string game_pretext = line.substr(0, pos);
            std::cout << game_pretext << std::endl;

            // Within "Game x" --> x
            auto pos_game_idx = game_pretext.find(" ");
            game_vals.w = std::atoi(game_pretext.substr(pos_game_idx + 1, game_pretext.size() - pos_game_idx).c_str()) - 1;
            
            // Everything after "Game x:"
            line.erase(0, pos + delimiter.length());
            delimiter = ";";
            num_games++;

            while (pos != std::string::npos)
            {
                pos = line.find(delimiter);
                std::string game_instance = line.substr(0, pos);
                std::cout << game_instance << std::endl;
                line.erase(0, pos + delimiter.length());

                std::string delim_within_instance = ",";
                size_t posi = 0; 
                while (posi != std::string::npos)
                {
                    posi = game_instance.find(delim_within_instance);
                    std::string game_instance_value = game_instance.substr(0, posi);
                    std::cout << game_instance_value << std::endl;
                    game_instance.erase(0, posi + delimiter.length());

                    size_t pos_color = 0;
                    pos_color = game_instance_value.find(" ");
                    game_instance_value.erase(0, pos_color + 1);
                    pos_color = game_instance_value.find(" ");
                    auto num = game_instance_value.substr(0, pos_color);
                    auto color = game_instance_value.substr(pos_color + 1, game_instance_value.size());

                    if (color == "red")
                    {
                        game_vals.x = std::atoi(num.c_str());
                    }
                    else if (color == "green")
                    {
                        game_vals.y = std::atoi(num.c_str());
                    }
                    else
                    {
                        game_vals.z = std::atoi(num.c_str());
                    }
                }

                std::cout << (int)game_vals.x << ", " << (int)game_vals.y << ", " << (int)game_vals.z << ", " << (int)game_vals.w << std::endl;
                combos.push_back(game_vals);
            }
        }
    }

    // Allocate data on device and copy it over
    void* cuCombos = nullptr;
    void* cuResult = nullptr;
    checkCUDAError(cudaMalloc(&cuCombos, sizeof(uchar4) * combos.size())); // in
    checkCUDAError(cudaMalloc(&cuResult, sizeof(uint32_t)));               // out, single num

    checkCUDAError(cudaMemcpy(cuCombos, combos.data(), combos.size() * sizeof(uchar4), cudaMemcpyHostToDevice));

    // Run problem on device
    advc_problem_2((uchar4*)cuCombos, (uint32_t*)cuResult, combos.size(), num_games);

    // Copy over data from device to host
    uint32_t result = 0;
    checkCUDAError(cudaMemcpy(&result, cuResult, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Assert results
    std::cout << "Valid game id total = " << result << ", correct result = " << 2237 << std::endl;
    assert(result == 2237);
}

void runProblem_1(std::string inputPath, bool part_one = true)
{
    // Read problem data as:
    // tokens:      ascii counterpart of every character, excluding newlines
    // endline_idx: index of the last character of everyline. The indices are w.r.t the tokens array

    // The way the problem is solved is by doing a reduction, sigh
    std::ifstream input_file(inputPath);
    std::vector<unsigned char> tokens;
    tokens.reserve(10000);
    std::vector<short> endline_idx;
    int max_line_size = 0;
    if (input_file.is_open())
    {
        size_t current_size = 0;
        while (input_file.good())
        {
            std::string line;
            input_file >> line;
            if (line == "")
            {
                continue;
            }

            max_line_size = std::max(max_line_size, (int)line.length());

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
    if (part_one)
    {
        advc_problem_1_1((uchar*)cuTokens, (short*)cuIndices, (uint32_t*)cuSum, tokens.size(), endline_idx.size(), max_line_size);
    }
    else
    {
        advc_problem_1_2((uchar*)cuTokens, (short*)cuIndices, (uint32_t*)cuSum, tokens.size(), endline_idx.size(), max_line_size);
    }

    // Copy over data from device to host
    uint32_t sum = 0;
    checkCUDAError(cudaMemcpy(&sum, cuSum, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Assert results
    if (part_one)
    {
        std::cout << "Sum = " << sum << ", correct sum = " << 55108 << std::endl;
        assert(sum == 55108);
    }
    else
    {
        std::cout << "Sum = " << sum << ", correct sum = " << 56324 << std::endl;
        assert(sum == 56324);
    }
}

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("AdventOfCode2023");

    program.add_argument("--problem", "-p")
        .help("Runs a given Advent Of Code 2023 problem")
        .required()
        .default_value(1)
        .scan<'f', float>();

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
        auto pb = program.get<float>("--problem");
        if (pb == 1.1f)
        {
            runProblem_1(in_path);
        }
        else if (pb == 1.2f)
        {
            runProblem_1(in_path, false);
        }
        else if (pb == 2.0f)
        {
            runProblem_2(in_path);
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
