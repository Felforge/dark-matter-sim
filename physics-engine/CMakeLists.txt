cmake_minimum_required(VERSION 3.18) # Minimum CMake version
project(DarkMatterNBody LANGUAGES CXX CUDA) # Define the project, enabling CUDA support

# Set output directory for static libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Enable CUDA support
find_package(CUDA REQUIRED)

# Define the static library
add_library(barnes_hut STATIC cuda/barneHut.cu)

# Set properties for the static library
set_target_properties(barnesHut PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # Allow separate compilation of CUDA files
    POSITION_INDEPENDENT_CODE ON   # Required for static libraries
)

# Link CUDA runtime
target_link_libraries(barnes_hut CUDA::cudart)
