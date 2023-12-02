# CUDA lib settings
add_library(cuda_kernels STATIC
	kernels.cuh
	kernels.cu)
	
# EXE settings
add_executable(cuda_playpen main.cpp)

message("${CMAKE_CURRENT_LIST_DIR}")
message("${ARGPARSE_INCLUDE_DIR}")

include_directories("${CMAKE_CURRENT_LIST_DIR}" "${ARGPARSE_INCLUDE_DIR}")
set_target_properties(cuda_playpen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_playpen PRIVATE cuda_kernels)
