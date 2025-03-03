#include <cuda_runtime.h>
#include <cmath>


// Copied from go program
struct Particle {
    double Mass;
    double X, Y, Z;
    double Vx, Vy, Vz;
    double Ax, Ay, Az;
};

// atomicAdd does not support doubles by default so this is needed
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void accumulateItemsKernel(Particle* particles, int numParticles, double* dA, double* dD) {
    // Calculate the global thread index
    // Each thread gets a unique "idx" to work on a different particle
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread index is within bounds
    if (idx >= numParticles) {
        return;
    }
    Particle& p = particles[idx];

    double magnitudeA = sqrt(p.Ax * p.Ax + p.Ay * p.Ay + p.Az * p.Az);
    atomicAddDouble(dA, magnitudeA);

    double accumulatedDistance = 0.0;
    for (int i = 0; i < numParticles; i++) {
        if (i == idx) {
            continue;
        }
        double dx = p.X - particles[i].X;
        double dy = p.Y - particles[i].Y;
        double dz = p.Z - particles[i].Z;
        accumulatedDistance += sqrt(dx * dx + dy * dy + dz * dz);
    }
    atomicAddDouble(dD, accumulatedDistance / (numParticles - 1));
}

extern "C" {
    __declspec(dllexport) double getTimeStep(Particle* particles, int numParticles, double timeStepParameter, double softeningDivisor) {
        // Assign number of threads
        int threads = 256;
        // Compute the number of thread blocks needed
        int blocks = (numParticles + threads - 1) / threads;

        // Create device variables
        double *dA, *dD;

        // Allocate memory
        cudaMalloc(&dA, sizeof(double));
        cudaMalloc(&dD, sizeof(double));

        // Set values
        cudaMemset(dA, 0, sizeof(double));
        cudaMemset(dD, 0, sizeof(double));
        
        // Accumulate distances and accelerations in parallel on the GPU
        accumulateItemsKernel<<<blocks, threads>>>(particles, numParticles, dA, dD);

        // Wait for everything to finish
        cudaDeviceSynchronize();

        // Move items to host
        double hA, hD;
        cudaMemcpy(&hA, &dA, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&hD, &dD, sizeof(double), cudaMemcpyDeviceToHost);

        // Free device variables from memory
        cudaFree(dA);
        cudaFree(dD);

        // Average out
        hA /= numParticles;
        hD /= numParticles;

        // Solve for softening length
        double softeningLength = hD / softeningDivisor;

        // Compute time step
        double timeStep = timeStepParameter * sqrt(softeningLength / hA);

        // Return time step
        return timeStep;
    }
}