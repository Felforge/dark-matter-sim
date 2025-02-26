// #include "../headers/BarnesHut.h"
// #include "../cuda/BarnesHut.cu"
// #include "cuda_runtime.h"

// // Create a Barnes-Hut instance
// extern "C" void* createBarnesHut(int numBodies, Particle* particles, double bounds[3][2]) {
//     return new BarnesHut(numBodies, particles, bounds);
// }

// // Compute forces
// extern "C" void computeForces(void* tree, double theta, double mode) {
//     BarnesHut* barnesHut = (BarnesHut*)tree;
//     barnesHut->computeForces(theta, mode);
//     cudaDeviceSynchronize();
// }

// // Free the Barnes-Hut instance
// extern "C" void destroyBarnesHut(void* tree) {
//     delete (BarnesHut*)tree;
// }

// // Name explains everything
// extern "C" void copyParticlesToHost(void* tree, Particle* hostParticles, int numBodies) {
//     BarnesHut* barnesHut = (BarnesHut*)tree;
//     cudaMemcpy(hostParticles, barnesHut->dBodies, numBodies * sizeof(Particle), cudaMemcpyDeviceToHost);
// }