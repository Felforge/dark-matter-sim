#include <cuda_runtime.h>
#include <cmath>

// Copied from go program
struct Particle {
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
};

__device__ double* dA = 0.0;
__device__ double* dD = 0.0;

__global__ void accumulateItemsKernel(Particle* particles, int numParticles) {
    // Calculate the global thread index
    // Each thread gets a unique "idx" to work on a different particle
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread index is within bounds
    if (idx >= numParticles) {
        return;
    }
    Particle& p = particles[idx];

    double magnitudeA = sqrt(p.ax * p.ax + p.ay * p.ay + p.az * p.az);
    atomicAdd(&dA, magnitudeA);

    double accumulatedDistance = 0.0;
    for (int i = 0; i < numParticles; i++) {
        if (i == idx) {
            continue;
        }
        double dx = p.x - particles[i].x;
        double dy = p.y - particles[i].y;
        double dz = p.z - particles[i].z;
        accumulatedDistance += sqrt(dx * dx + dy * dy + dz * dz);
    }
    atomicAdd(&dD, accumulatedDistance / (numParticles - 1));
}

double getTimeStep(Particle* particles, int numParticles, double timeStepParameter, double softeningDivisor) {
    // Assign number of threads
    int threads = 256;
    // Compute the number of thread blocks needed
    int blocks = (numParticles + threads - 1) / threads;
    
    // Accumulate distances and accelerations in parallel on the GPU
    accumulateItemsKernel<<<blocks, threads>>>(particles, numParticles);

    // Wait for everything to finish
    cudaDeviceSynchronize();

    // Move items to host
    double hA, hD;
    cudaMemcpy(&hA, &dA, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hD, &dD, sizeof(double), cudaMemcpyDeviceToHost);

    double softeningLength = hD / softeningDivisor;
    return timeStepParameter * sqrt(softeningLength / hA);
}