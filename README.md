# cuda-playpen
I'm using this repo to upload various fun experiments with CUDA.

## Advent of Code 2023
I'll be solving Adv of Code problems using CUDA.

### Building (on Windows)

#### Requirements
You can probably make this work on older stuff, but here's what I used:
* You can find the latest CUDA SDK [here](https://developer.nvidia.com/cuda-downloads)
* Visual Studio 19

#### Building
 Generating a Visual Studio solution
```
mkdir build
cd build
cmake ..
```

Open the generated solution in `build` then compile everything. You can run the `cuda_playpen.exe` with the following args to run the CUDA solution of a particular **Adv of Code 2023** problem:

`cuda_playpen.exe --problem 1 --path path-to-input-data`
