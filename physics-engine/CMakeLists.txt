cmake_minimum_required(VERSION 3.18) # Minimum CMake version
project(DarkMatterNBody CUDA) # Define the project, enabling CUDA support

# Set output directory for shared libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Enable CUDA support
find_package(CUDA REQUIRED)

# All libraries below
add_library(barnes_hut SHARED cuda/barnes_hut.cu)

set_target_properties(barnes_hut PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # Allow separate compilation of CUDA files
    POSITION_INDEPENDENT_CODE ON   # Required for shared libraries
)

# Link CUDA runtime
target_link_libraries(barnes_hut CUDA::cudart)