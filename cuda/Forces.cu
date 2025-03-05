#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define G 6.67428e-11f

// Copied from go program
struct Particle {
    double Mass;
    double X, Y, Z;
    double Vx, Vy, Vz;
    double Ax, Ay, Az;
};

// Define CUDA_CHECK macro to check for CUDA errors
// Used over like every CUDA function to catch possible errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

//Everything is pre-declared to avoid errors
class BarnesHut;
__global__ void createChildrenKernel(Particle* inpBodies, int inpNumBodies, Particle* octantLists[8], int* octantCounts, double mid[3]);
__global__ void computeForcesKernel(Particle* inpBodies, BarnesHut* tree, double theta, double mode=1.0);

class BarnesHut {
    public:
        // Checking if top layer
        int totalBodies;

        // Total mass
        double mass;

        // Center of mass
        double xcm;
        double ycm;
        double zcm;

        int numBodies;
        double bounds[3][2]; // 3x2 Matrix
        BarnesHut* children[8];

        // Needed for checking if everything is on the same point
        double minDistance;

        // For CUDA stuff
        double octantBounds[8][3][2];
        Particle* dBodies;
        int* dOctantCounts = nullptr; // initialize as nullptr
        Particle** dOctantLists = nullptr; // initialize as nullptr

        // Constructor
        // dBodies will be edited or just used by everything it is passed to
        // if total bodies is 0 it is set to the number of bodies
        BarnesHut(int inpNumBodies, Particle* inpDBodies, double inpBounds[3][2], int inpTotalBodies=0) {
            // Set dBodies as a pointer to inpDBodies
            dBodies = inpDBodies;

            // Set total bodies
            if (inpTotalBodies == 0) {
                totalBodies = inpNumBodies;
            } else {
                totalBodies = inpTotalBodies;
            }

            // Copy numBodies
            numBodies = inpNumBodies;

            // Copy bounds
            for (int i = 0; i < 3; i++) {
                bounds[i][0] = inpBounds[i][0];
                bounds[i][1] = inpBounds[i][1];
            }

            // Set zero mass and center of mass
            mass = 0.0;
            xcm = ycm = zcm = 0.0;

            // Initialize children pointers to nullptr
            for (int i = 0; i < 8; i++) {
                children[i] = nullptr;
            }

            // Call tree construction
            constructTree(dBodies, inpNumBodies);
        }

        // Destructor
        ~BarnesHut() {
            // Free GPU memory from dBodies
            for (int i = 0; i < 8; i++) {
                if (children[i] != nullptr) {
                    delete children[i];
                    children[i] = nullptr;
                }
            }

            // Memory was allocated outside
            // Dont free top layer
            // Make sure everything wasn't freed already
            if (numBodies != totalBodies && dBodies != nullptr) {
                CUDA_CHECK(cudaFree(dBodies));
                dBodies = nullptr;
            }
            if (dOctantCounts != nullptr) {
                CUDA_CHECK(cudaFree(dOctantCounts));
                dOctantCounts = nullptr;
            }
            if (dOctantLists != nullptr) {
                for (int i = 0; i < 8; i++) {
                    if (dOctantLists[i] != nullptr) {
                        CUDA_CHECK(cudaFree(dOctantLists[i]));
                        dOctantLists[i] = nullptr;
                    }
                }
                CUDA_CHECK(cudaFree(dOctantLists));
                dOctantLists = nullptr;
            }
        }

        // Computer forces using parallelization
        // See internal function comment for mode
        void computeForces(double theta=0.5, double mode=1.0) {
            // Assign number of threads
            int threads = 256;
            // Compute the number of thread blocks needed
            int blocks = (numBodies + threads - 1) / threads;
            
            // Calculate forces in parallel on the GPU
            computeForcesKernel<<<blocks, threads>>>(dBodies, this, theta, mode);

            // Wait for the octree construction to finish
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    private:
        bool allSamePoint(Particle* inpBodies, int inpNumBodies, double minDistance=1e-50) {
            Particle first = inpBodies[0];
            for (int i = 1; i < inpNumBodies; i++) {
                if (fabs(inpBodies[i].X - first.X) > minDistance ||
                    fabs(inpBodies[i].Y - first.Y) > minDistance ||
                    fabs(inpBodies[i].Z - first.Z) > minDistance) {
                        return false;
                    }
            }
            return true;
        }

        void constructTree(Particle* inpBodies, int inpNumBodies) {
            if (inpNumBodies == 0) {
                return;
            }
            if (allSamePoint(inpBodies, inpNumBodies)) {
                for (int i = 0; i < inpNumBodies; i++) {
                    mass += inpBodies[i].Mass;
                }
                
                // Center of mass of all will be the same
                xcm = inpBodies[0].X;
                ycm = inpBodies[0].Y;
                zcm = inpBodies[0].Z;

            } else {
                createChildren();
            }
        }

        // Create children subtree
        void createChildren() {
            // Split into octants
            double mid[3];
            for (int i = 0; i < 3; i++) {
                mid[i] = (bounds[i][0] + bounds[i][1]) / 2;
            }

            // Define new bounds of octants
            for (int i = 0; i < 8; i++) { 
                for (int j = 0; j < 3; j++) {
                    octantBounds[i][j][0] = ((i >> j) & 1) ? mid[j] : bounds[j][0];
                    octantBounds[i][j][1] = ((i >> j) & 1) ? bounds[j][1] : mid[j];
                }
            }

            // Allocate memory for octant lists
            CUDA_CHECK(cudaMallocManaged(&dOctantLists, 8 * sizeof(Particle*)));
            for (int i = 0; i < 8; i++) {
                CUDA_CHECK(cudaMallocManaged(&dOctantLists[i], numBodies * sizeof(Particle)));
            }

            // Allocate memory for counters
            CUDA_CHECK(cudaMallocManaged(&dOctantCounts, 8 * sizeof(int)));
            CUDA_CHECK(cudaMemset(dOctantCounts, 0, 8 * sizeof(int)));

            // Launch CUDA Kernel
            int threads = 256;
            int blocks = (numBodies + threads - 1) / threads;
            createChildrenKernel<<<blocks, threads>>>(dBodies, numBodies, dOctantLists, dOctantCounts, mid);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Create children instances
            for (int i = 0; i < 8; i++) {
                children[i] = new BarnesHut(dOctantCounts[i], dOctantLists[i], octantBounds[i], totalBodies);
            }
        }
};

// Needed inside and outside the class
__host__ __device__ double getDistance(double x1, double y1, double z1, double x2, double y2, double z2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// CUDA Kernel to sort particles
__global__ void createChildrenKernel(Particle* inpBodies, int inpNumBodies, Particle* octantLists[8], int* octantCounts, double mid[3]) {
    // Calculate the global thread index
    // Each thread gets a unique "idx" to work on a different particle
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread index is within bounds
    if (idx >= inpNumBodies) {
        return;
    }
    Particle& particle = inpBodies[idx];

    int octant = 0;
    if (particle.X > mid[0]) {
        // Set first bit to 1
        octant |= 1;
    }
    if (particle.Y > mid[1]) {
        // Set second bit to 1
        octant |= 2;
    }
    if (particle.Z > mid[2]) {
        // Set 3rd bit to 1
        octant |= 4;
    }

    // Get index for this particle in the octant list
    int insertIdx = atomicAdd(&octantCounts[octant], 1);

    // Prevent out of bounds access
    if (insertIdx >= inpNumBodies) {
        printf("Out of bounds access: octant %d, index %d (numBodies: %d)\n", octant, insertIdx, inpNumBodies);
        return;
    }

    octantLists[octant][insertIdx] = particle;
}

// Mode takes either 1.0 (attract) or -1.0 (repel)
// Inverse Gravitational Force is needed for Glass Configuartion
__global__ void computeForcesKernel(Particle* inpBodies, BarnesHut* tree, double theta, double mode) {
    // Calculate the global thread index
    // Each thread gets a unique "idx" to work on a different particle
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread index is within bounds
    if (idx >= tree->numBodies) {
        return;
    }
    Particle& p = inpBodies[idx];

    // Declare empty acceleration vector fields
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    // Set an explicit stack depth of 128 (even this should be way too much)
    const int MAX_STACK_SIZE = 128;
    BarnesHut* stack[MAX_STACK_SIZE];
    int stackTop = 0;

    // Start stack with tree root node
    stack[stackTop++] = tree;

    // Iterate through whole stack
    while (stackTop > 0) {
        // Boundary condition
        if (stackTop >= MAX_STACK_SIZE) {
            printf("Stack overflow for particle %d\n", idx);
            return;
        }

        // Pop a node from the stack
        BarnesHut* node = stack[--stackTop];

        // Catch zero cases
        if (node->mass == 0 || (node->xcm == p.X && node->ycm == p.Y && node->zcm == p.Z)) {
            continue;
        }

        double d = getDistance(p.X, p.Y, p.Z, node->xcm, node->ycm, node->zcm);
        double s = node->bounds[0][1] - node->bounds[0][0];

        // Barnes-Hut assumes sufficiently far away enough particles to be in larger nodes
        // The condition that needs to be met is s / d < theta
        if ((s / d < theta) || node->numBodies == 1) {
            double a = (mode * G * node->mass) / (d * d);
            ax += a * (node->xcm - p.X) / d;
            ay += a * (node->ycm - p.Y) / d;
            az += a * (node->zcm - p.Z) / d;
        } else {
            // Push all nodes that do that satisfy the above 
            // condition further into the stack for processing
            if (node->children[0] != nullptr) {
                // If the first item is null all others must be
                for (int i = 0; i < 8; i++) {
                    stack[stackTop++] = node->children[i];
                }
            }
        }
    }
    // Add acceleration into particle
    p.Ax = ax;
    p.Ay = ay;
    p.Az = az;
}

// Yoshida constants
const double D1 = 1.0 / (2.0 - cbrt(2.0));
const double D2 = -1.0 * cbrt(2.0) / (2.0 - cbrt(2.0));
const double C1 = D1 / 2.0;
const double C2 = (D1 + D2) / 2.0;

// Kernel to update position
// Each step has a slightly different multiplier
__global__ void yoshidaPositionKernel(int numParticles, Particle* particles, double distance, double dt, double multiplier) {
    // Calculate the global thread index
    // Each thread gets a unique "idx" to work on a different particle
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread index is within bounds
    if (idx >= numParticles) {
        return;
    }
    Particle& p = particles[idx];

    // Apply formula
    // Make sure particles stay in the simulation box
    double x = p.X + multiplier * p.Vx * dt;
    if (x > distance) {
        p.X = distance;
    } else if (x < 0.0) {
        p.X = 0.0;
    } else {
        p.X = x;
    }

    double y = p.Y + multiplier * p.Vy * dt;
    if (y > distance) {
        p.Y = distance;
    } else if (y < 0.0) {
        p.Y = 0.0;
    } else {
        p.Y = y;
    }

    double z = p.Z + multiplier * p.Vz * dt;
    if (z > distance) {
        p.Z = distance;
    } else if (z < 0.0) {
        p.Z = 0.0;
    } else {
        p.Z = z;
    }
}

// Kernel to update velocity
// Each step has a slightly different multiplier
__global__ void yoshidaVelocityKernel(int numParticles, Particle* particles, double dt, double multiplier) {
    // Calculate the global thread index
    // Each thread gets a unique "idx" to work on a different particle
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread index is within bounds
    if (idx >= numParticles) {
        return;
    }
    Particle& p = particles[idx];

    // Apply formula
    p.Vx += multiplier * p.Ax * dt;
    p.Vy += multiplier * p.Ay * dt;
    p.Vz += multiplier * p.Az * dt;
}

// Step Yoshida Position
void stepYoshidaPosition(int numParticles, Particle* particles, double distance, double dt, double multiplier, int threads, int blocks) {
    // Run Kernel and wait for everything to finish
    yoshidaPositionKernel<<<blocks, threads>>>(numParticles, particles, distance, dt, multiplier);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Step Yoshida Velocity
void stepYoshidaVelocity(int numParticles, Particle* particles, double bounds[3][2], double dt, double multiplier, double theta, double mode, int threads, int blocks) {
    // BarnesHut takes the host particles object
    BarnesHut* tree = new BarnesHut(numParticles, particles, bounds);
    tree->computeForces(theta, mode);

    // Delete tree
    delete tree;

    // Run Kernel and wait for everything to finish
    // The tree updated particles so it can be used
    yoshidaVelocityKernel<<<blocks, threads>>>(numParticles, particles, dt, multiplier);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Use Yoshida integration to time evolve the particles
// Doesn't return new list but updates inputted particles
// Distance is edge length of simulation box
// Needs to be C to be Go-compatible
extern "C" { 
    __declspec(dllexport) void applyYoshida(int numParticles, Particle* particles, double distance, double dt, double theta, double mode) {
        // Copy particles to device
        Particle* dParticles;
        CUDA_CHECK(cudaMalloc(&dParticles, numParticles * sizeof(Particle)));
        CUDA_CHECK(cudaMemcpy(dParticles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice));

        // Create bounds and copy to device
        double bounds[3][2];
        for (int i = 0; i < 3; i++) {
            bounds[i][0] = 0.0;
            bounds[i][1] = distance;
        }

        // Assign number of threads
        int threads = 256;
        // Compute the number of thread blocks needed
        int blocks = (numParticles + threads - 1) / threads;

        // Apply Yoshida
        // Synchronization is done after everything inside the function

        // Step 1 Position
        stepYoshidaPosition(numParticles, dParticles, distance, dt, C1, threads, blocks);

        // Step 1 Velocity
        stepYoshidaVelocity(numParticles, dParticles, bounds, dt, D1, theta, mode, threads, blocks);

        // Step 2 Position
        stepYoshidaPosition(numParticles, dParticles, distance, dt, C2, threads, blocks);

        // Step 2 Velocity
        stepYoshidaVelocity(numParticles, dParticles, bounds, dt, D2, theta, mode, threads, blocks);

        // Step 3 Position
        stepYoshidaPosition(numParticles, dParticles, distance, dt, C2, threads, blocks);

        // Step 3 Velocity
        stepYoshidaVelocity(numParticles, dParticles, bounds, dt, D1, theta, mode, threads, blocks);

        // Step 4 Position
        stepYoshidaPosition(numParticles, dParticles, distance, dt, C1, threads, blocks);

        // Copy dParticles back to the CPU
        cudaMemcpy(particles, dParticles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

        // Free dParticles from memory
        CUDA_CHECK(cudaFree(dParticles));
    }

    // Update particle accelerations based on the forces
    // Needed for very first time step
    __declspec(dllexport) void applyForces(int numParticles, Particle* particles, double distance, double theta, double mode) {
        // Copy particles to device
        Particle* dParticles;
        CUDA_CHECK(cudaMalloc(&dParticles, numParticles * sizeof(Particle)));
        CUDA_CHECK(cudaMemcpy(dParticles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice));

        // Create bounds
        double bounds[3][2];
        for (int i = 0; i < 3; i++) {
            bounds[i][0] = 0.0;
            bounds[i][1] = distance;
        }

        // Create Barnes-Hut tree and calculate particle accelerations
        BarnesHut* tree = new BarnesHut(numParticles, dParticles, bounds);
        tree->computeForces(theta, mode);

        // Delete tree
        delete tree;

        // Copy final particles back to host
        CUDA_CHECK(cudaMemcpy(particles, dParticles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost));

        // Free dParticles from memory
        CUDA_CHECK(cudaFree(dParticles));
    }
}
