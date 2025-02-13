#include <cuda_runtime.h>
#include <cmath>
#include <string>

#define G 6.67430e-11f

// Copied from rust program
struct Particle {
    double* mass;
    double* x, *y, *z;
    double* vx, *vy, *vz;
    double* ax, *ay, *az;
};

class BarnesHut {
    public:
        double mass = 0;
        // Center of mass
        double xcm;
        double ycm;
        double zcm;

        // Constructor
        BarnesHut(int inpNumBodies, Particle* inpBodies, double inpBounds[3][2], int inpThreads) {
            numBodies = inpNumBodies;
            for (int i = 0; i < 3; i++) {
                bounds[i][0] = inpBounds[i][0];
                bounds[i][1] = inpBounds[i][1];
            }
            mass = 0;
            xcm = ycm = zcm = 0;

            // Allocate GPU memory to dBodies
            cudaMallocManaged(&dBodies, numBodies * sizeof(Particle));

            // Copy Bodies from CPU to GPU
            cudaMemcpy(dBodies, inpBodies, numBodies * sizeof(Particle), cudaMemcpyHostToDevice);

            // Number of threads per block
            threads = inpThreads;

            // Compute the number of thread blocks needed
            int blocks = (numBodies + threads - 1) / threads;

            // Construct the octree in parallel on the GPU
            // Each thread assigns a body to the tree using atomic operations
            constructTree<<<blocks, threads>>>(dBodies, this, numBodies); // Function not made yet

            // Wait for the octree construction to finish
            cudaDeviceSynchronize(); 

            // Normalize center of mass
            // The mass was accumulated by "construct_tree"
            xcm /= mass;
            ycm /= mass;
            zcm /= mass;
        }

        // Destructor
        ~BarnesHut() {
            // Free GPU memory from dBodies
            cudaFree(dBodies);
        }

        // Computer forces using parallelization
        // See internal function comment for mode
        void computeForces(double theta=0.5, double mode=1.0) {
            // Compute the number of thread blocks needed
            int blocks = (numBodies + threads - 1) / threads;
            
            // Calculate forces in parallel on the GPU
            computeForcesInternal<<blocks, threads>>(bodies, this, numBodies, theta, mode);

            // Wait for the octree construction to finish
            cudaDeviceSynchronize();

            // Copy Data from GPU to CPU
            // May or may not be needed
            cudaMemcpy(bodies, dBodies, numBodies * sizeof(Particle), cudaMemcpyDeviceToHost);
        }

    private:
        int threads;
        int numBodies;
        Particle* bodies;
        double bounds[3][2]; // 3x2 Matrix
        BarnesHut* children[8];

        // Needed for checking if everything is on the same point
        double minDistance;

        // Needed to check if everything is on the same point (edge case)
        bool allSamePoint;

        __device__ double getDistance(double x1, double y1, double z1, double x2, double y2, double z2) {
            double dx = x2 - x1;
            double dy = y2 - y1;
            double dz = z2 - z1;
            return sqrt(dx * dx + dy * dy + dz * dz);
        }

        __global__ void constructTree(Particle* inpBodies, BarnesHut* tree, int inpNumBodies) {
            // Calculate the global thread index
            // Each thread gets a unique "idx" to work on a different particle
            int idx = threadIdx.x + blockIdx.x * blockDim.x;

            // Ensure the thread index is within bounds
            if (idx >= inpNumBodies) {
                return;
            }

            // Retrieve the particle this thread is responsible for
            Particle& particle = inpBodies[idx];
            
            if (tree->numBodies > 1) {
                // Split into octants
                double xMid = (tree->bounds[0][0] + tree->bounds[0][1]) / 2;
                double yMid = (tree->bounds[1][0] + tree->bounds[1][1]) / 2;
                double zMid = (tree->bounds[2][0] + tree->bounds[2][1]) / 2;

                int octant = 0;
                if (particle.x > xMid) {
                    // Set first bit to 1
                    octant |= 1;
                }
                if (particle.y > tree->ycm) {
                    // Set second bit to 1
                    octant |= 2;
                }
                if (particle.z > tree->zcm) {
                    // Set 3rd bit to 1
                    octant |= 4;
                }

                if (tree->children[octant] == nullptr) {
                    double newBounds[3][2];

                    // Subdivide bounds
                    for (int i = 0; i < 3; i++) {
                        double mid = (tree->bounds[i][0] + tree->bounds[i][1]) / 2;
                        newBounds[i][0] = (octant & (1 << i)) ? mid : tree->bounds[i][0];
                        newBounds[i][1] = (octant & (1 << i)) ? tree->bounds[i][1] : mid;
                    } 
                    // CONTINUE BY SUBDIVIDING BODIES BY BOUNDS
                    // Maybe look into reworking this function completely logic seems off
                }

            } else if (tree->numBodies == 1) {
                // Update total mass
                // atomicAdd is needed as multiple threads are adding at once
                atomicAdd(&tree->mass, particle.mass);

                // Add to center of mass
                // Must be divided later by total mass
                atomicAdd(&tree->xcm, particle.mass * particle.x);
                atomicAdd(&tree->ycm, particle.mass * particle.y);
                atomicAdd(&tree->zcm, particle.mass * particle.z);
            }
        }

        // Mode takes either 1.0 (attract) or -1.0 (repel)
        // Inverse Gravitational Force is needed for Glass Configuartion
        __global__ void computeForcesInternal(Particle* inpBodies, BarnesHut* tree, int inpBodyCount, double theta, double mode=1.0) {
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
                    ax += a * (xcm - node->x) / d;
                    ay += a * (ycm - node->y) / d;
                    az += a * (zcm - node->z) / d;
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
};