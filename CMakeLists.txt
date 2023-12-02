cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

project(cuda-playpen)

enable_language(CXX)
enable_language(CUDA)

set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")
set(ARGPARSE_INCLUDE_DIR "${EXTERNAL}/argparse")

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)
