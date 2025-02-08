#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

// NOT WORKING

// Solve Poisson Equation
extern "C" __global__
void poisson_solver(cufftComplex* delta_k, cufftComplex* phi_k, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N * N) return;

    int kx = (idx / (N * N)) % N;
    int ky = (idx / N) % N;
    int kz = idx % N;

    float kx_f = (kx < N / 2) ? kx : kx - N;
    float ky_f = (ky < N / 2) ? ky : ky - N;
    float kz_f = (kz < N / 2) ? kz : kz - N;

    float k2 = kx_f * kx_f + ky_f * ky_f + kz_f * kz_f;
    if (k2 > 0) {
        phi_k[idx].x = delta_k[idx].x / k2;
        phi_k[idx].y = delta_k[idx].y / k2;
    } else {
        phi_k[idx].x = phi_k[idx].y = 0.0f;
    }
}

// Compute Displacement Field
extern "C" __global__
void compute_displacement(cufftComplex* phi_k, float* s, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N * N) return;

    int kx = (idx / (N * N)) % N;
    int ky = (idx / N) % N;
    int kz = idx % N;

    float2 phi = phi_k[idx];
    s[idx] = -phi.y * kx + -phi.x * ky + -phi.y * kz;
}
