#include <cuda_runtime.h>
#include <cmath>

#define G 6.67430e-11f

// Copied from rust program
struct Particle {
    double mass;
    double position[3];
    double velocity[3];
};

class Octree {
    public:
        int numBodies;
        Particle[numBodies] bodies;
        double boxSize;
        double mass = 0;
        double[3] cm = [0.0, 0.0, 0.0];

        // Needed to check if particles are in the same position
        double minDistance;

        __host__ __device__ Octree(Particle[numBodies] inpBodies, int inpNumBodies, double inpBoxSize, double inpMinDistance=1e-10) {
            bodies = inpBodies;
            numBodies = inpNumBodies;
            boxSize = inpBoxSize;
            minDistance = inpMinDistance;

            if (numBodies > 0) {
                firstX = bodies[0].position[0]
                firstY = bodies[0].position[1]
                firstZ = bodies[0].position[2]
                allSamePoint = true;
                for 
            }
        }
}