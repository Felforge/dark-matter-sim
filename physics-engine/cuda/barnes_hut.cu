#include <cuda_runtime.h>
#include <cmath>
#include <string>

#define G 6.67430e-11f

// Copied from rust program
struct Particle {
    double mass;
    double position[3];
    double velocity[3];
    double acceleration[3];
};

class BarnesHut {
    public:
        double mass = 0;
        double cm[3]; // Center of mass

        // Constructor
        BarnesHut(int inpNumBodies, Particle* inpBodies, double inpBounds[3][2]) {
            numBodies = inpNumBodies;
            for (int i = 0; i < 3; i++) {
                bounds[i][0] = inpBounds[i][0];
                bounds[i][1] = inpBounds[i][1];
            }
            mass = 0;
            cm[0] = cm[1] = cm[2] = 0;

            // Allocate GPU memory to dBodies
            cudaMallocManaged(&dBodies, numBodies * sizeof(Particle));

            // Copy Bodies from CPU to GPU
            cudaMemcpy(dBodies, inpBodies, numBodies * sizeof(Particle), cudaMemcpyHostToDevice);

            // Number of threads per block
            int threads = 256;

            // Computer the number of thread blocks needed
            int blocks = (numBodies + threads - 1) / threads;

            // Construct the octree in parallel on the GPU
            // Each thread assigns a body to the tree using atomic operations
            constructTree<<<blocks, threads>>>(dBodies, this, numBodies); // Function not made yet

            // Wait for the octree construction to finish
            cudaDeviceSynchronize(); 

            // Normalize center of mass
            // The mass was accumulated by "construct_tree"
            for (int i = 0; i < 3; i++) {
                cm[i] /= mass;
            }
        }

        // Destructor
        ~BarnesHut() {
            // Free GPU memory from dBodies
            cudaFree(dBodies);
        }

    private:
        int numBodies;
        Particle* bodies;
        double bounds[3][2]; // 3x2 Matrix
        Octree* children[8];
        // Needed for checking if everything is on the same point
        double minDistance;
        // Needed to check if everything is on the same point (edge case)
        bool allSamePoint;

        __device__ double getDistance(double initial[3], double final[3]) {
            double dx = final[0] - initial[0];
            double dy = final[1] - initial[1];
            double dz = final[2] - initial[2];
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

            // Update total mass
            // atomicAdd is needed as multiple threads are adding at once
            atomicAdd(&tree->mass, body.m);

            // Add to center of mass
            // Must be divided later by total mass
            for (int i = 0; i < 3; i++) {
                atomicAdd(&tree->cm[i], body.cm[i]);
            }
        }

        // Mode takes either 1.0 (attract) or -1.0 (repel)
        // Inverse Gravitational Force is needed for Glass Configuartion
        __global__ void computeForces(Particle* inpBodies, BarnesHut* tree, int inpBodyCount, double theta, double mode=1.0) {
            // Calculate the global thread index
            // Each thread gets a unique "idx" to work on a different particle
            int idx = threadIdx.x + blockIdx.x * blockDim.x;

            // Ensure the thread index is within bounds
            if (idx >= tree->numBodies) {
                return;
            }
            Particle& p = inpBodies[idx];

            // Declare empty acceleration vector fields
            double a[3] = {0.0, 0.0, 0.0};

            // Set an explicit stack depth of 64 (even this should be way too much)
            BarnesHut* stack[64];
            int stackTop = 0;

            // Start stack with tree root node
            stack[stackTop++] = tree;

            // Iterate through whole stack
            while (stackTop > 0) {
                // Pop a node from the stack
                BarnesHut* node = stack[stackTop--];

                // Catch zero cases
                if (node->mass == 0 || (node->cm[0] == p.position[0] && node->cm[1] == b.position[1] && node->cm[2] == b.position[2])) {
                    continue;
                }

                double d = getDistance(p.position, cm);
                double s = node->bound[0][1] - node->bound[0][0];

                // Barnes-Hut assumes sufficiently far away enough particles to be in larger nodes
                // The condition that needs to be met is s / d < theta
                if ((s / d < theta) || node->numBodies == 1) {
                    double a = (mode * G * p.mass * node->mass) / (d * d * m);
                    for (int i = 0; i < 3; i++) {
                        a[i] += a * (cm[i] - node->position[i]) / d;
                    }
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
            p.acceleratin = a;
        }
}

// ADD THIS TYPE OF THING FOR FORCES
//int threads = 256;

            // Computer the number of thread blocks needed
            //int blocks = (numBodies + threads - 1) / threads;

            // Construct the octree in parallel on the GPU
            // Each thread assigns a body to the tree using atomic operations
            //constructTree<<<blocks, threads>>>(dBodies, this, numBodies); // Function not made yet