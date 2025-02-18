#include <cuda_runtime.h>
#include <cmath>

#define G 6.67430e-11f

// Copied from rust program
struct Particle {
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
};

//Everything is pre-declared to avoid errors
class BarnesHut;
__global__ void createChildrenKernel(Particle* inpBodies, int inpNumBodies, Particle* octantLists[8], int* octantCounts, double mid[3]);
__global__ void computeForcesKernel(Particle* inpBodies, BarnesHut* tree, double theta, double mode=1.0);

class BarnesHut {
    public:
        // Total mass
        double mass = 0;

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
        int* dOctantCounts;
        Particle** dOctantLists;

        // Constructor
        BarnesHut(int inpNumBodies, Particle* inpBodies, double inpBounds[3][2]) {
            numBodies = inpNumBodies;
            for (int i = 0; i < 3; i++) {
                bounds[i][0] = inpBounds[i][0];
                bounds[i][1] = inpBounds[i][1];
            }
            mass = 0;
            xcm = ycm = zcm = 0;

            constructTree(inpBodies, inpNumBodies);

            // Move tree to GPU after construction
            cudaMallocManaged(&dBodies, numBodies * sizeof(Particle));
            cudaMemcpy(dBodies, inpBodies, numBodies * sizeof(Particle), cudaMemcpyHostToDevice);
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
            cudaFree(dBodies);
            cudaFree(dOctantCounts);
            for (int i = 0; i < 8; i++) {
                cudaFree(dOctantLists[i]);
            }
            cudaFree(dOctantLists);
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
            cudaDeviceSynchronize();
        }

    private:
        bool allSamePoint(Particle* inpBodies, int inpNumBodies, double minDistance=1e-50) {
            Particle first = inpBodies[0];
            for (int i = 1; i < inpNumBodies; i++) {
                if (fabs(inpBodies[i].x - first.x) > minDistance &&
                    fabs(inpBodies[i].y - first.y) > minDistance &&
                    fabs(inpBodies[i].z - first.z) > minDistance) {
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
                    mass += inpBodies[i].mass;
                }
                
                // Center of mass of all will be the same
                xcm = inpBodies[0].x;
                ycm = inpBodies[0].y;
                zcm = inpBodies[0].z;

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
            cudaMallocManaged(&dOctantLists, 8 * sizeof(Particle*));
            for (int i = 0; i < 8; i++) {
                cudaMallocManaged(&dOctantLists[i], numBodies * sizeof(Particle));
            }

            // Allocate memory for counters
            cudaMallocManaged(&dOctantCounts, 8 * sizeof(int));
            cudaMemset(dOctantCounts, 0, 8 * sizeof(int));

            // Launch CUDA Kernel
            int threads = 256;
            int blocks = (numBodies + threads - 1) / threads;
            createChildrenKernel<<<blocks, threads>>>(dBodies, numBodies, dOctantLists, dOctantCounts, mid);
            cudaDeviceSynchronize();

            // Copy dOctantCounts to CPU
            int hOctantCounts[8];  // Host-side array
            cudaMemcpy(hOctantCounts, dOctantCounts, 8 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();  // Ensure copy is complete before using it

            // Create children instances
            Particle* hOctantList;
            for (int i = 0; i < 8; i++) {
                cudaMemcpy(hOctantList, dOctantLists[i], hOctantCounts[i] * sizeof(Particle), cudaMemcpyDeviceToHost);
                children[i] = new BarnesHut(hOctantCounts[i], dOctantLists[i], octantBounds[i]);
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
    if (particle.x > mid[0]) {
        // Set first bit to 1
        octant |= 1;
    }
    if (particle.y > mid[1]) {
        // Set second bit to 1
        octant |= 2;
    }
    if (particle.z > mid[2]) {
        // Set 3rd bit to 1
        octant |= 4;
    }

    // Get index for this particle in the octant list
    int insertIdx = atomicAdd(&octantCounts[octant], 1);

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
            return;
        }

        // Pop a node from the stack
        BarnesHut* node = stack[stackTop--];

        // Catch zero cases
        if (node->mass == 0 || (node->xcm == p.x && node->ycm == p.y && node->zcm == p.z)) {
            continue;
        }

        double d = getDistance(p.x, p.y, p.z, node->xcm, node->ycm, node->zcm);
        double s = node->bounds[0][1] - node->bounds[0][0];

        // Barnes-Hut assumes sufficiently far away enough particles to be in larger nodes
        // The condition that needs to be met is s / d < theta
        if ((s / d < theta) || node->numBodies == 1) {
            double a = (mode * G * node->mass) / (d * d);
            ax += a * (node->xcm - p.x) / d;
            ay += a * (node->ycm - p.y) / d;
            az += a * (node->zcm - p.z) / d;
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
    p.ax = ax;
    p.ay = ay;
    p.az = az;
}