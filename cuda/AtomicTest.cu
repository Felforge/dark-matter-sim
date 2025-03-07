#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel: Each thread atomically adds 1 to counts[0]
__global__ void testAtomicKernel(int* counts) {
    // All threads update counts[0]
    atomicAdd(&counts[0], 1);
}

int main() {
    const int arraySize = 8;
    int* d_counts;

    // Allocate 8 integers using unified (managed) memory
    CUDA_CHECK(cudaMallocManaged(&d_counts, arraySize * sizeof(int)));

    // Initialize all 8 elements to zero
    for (int i = 0; i < arraySize; i++) {
        d_counts[i] = 0;
    }

    // Set the number of threads
    const int threadsPerBlock = 256;
    const int blocks = 1;  // using a single block for simplicity

    // Launch the kernel.
    testAtomicKernel<<<blocks, threadsPerBlock>>>(d_counts);

    // Synchronize to ensure kernel has finished
    CUDA_CHECK(cudaDeviceSynchronize());

    // Expected value in d_counts[0] is threadsPerBlock * blocks
    printf("Final value in d_counts[0]: %d (expected %d)\n", d_counts[0], threadsPerBlock * blocks);

    // Free the unified memory.
    CUDA_CHECK(cudaFree(d_counts));

    return 0;
}
