#include <cuda_runtime.h>
#include <cmath>
#include <string>

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
        Particle* bodies;
        double mass = 0;
        double bounds[3][2]; // 3x2 Matrix
        double cm[3]; // Center of mass
        Octree* children[8];

        __host__ __device__ Octree(int inpNumBodies, Particle* inpBodies, double inpBounds[3][2], double inpMinDistance=1e-10) {
            numBodies = inpNumBodies;
            bodies = inpBodies;
            cm[0] = cm[1] = cm[2] = 0.0;
            mass = 0.0;
            minDistance = inpMinDistance;

            for (int i = 0; i < 3; i++) {
                bounds[i][0] = inpBounds[i][0];
                bounds[i][1] = inpBounds[i][1];
            }

            if (numBodies > 0) {
                if (checkAllSamePoint(numBodies, inpMinDistance)) {
                    for (int i = 0; i < numBodies; i++) {
                        mass += bodies[i].mass;
                        for (int j = 0; j < 3; j++) {
                            cm[j] += bodies[i].mass * bodies[i].position[j];
                        }
                    }
                    for (int j = 0; j < 3; j++) {
                        cm[j] /= mass;
                    }
                }
            }
            else {
                children = createChildren();
            }
        }
    private:
        // Needed for checking if everything is on the same point
        double minDistance;

        __host__ __device__ bool checkAllSamePoint(double inpMinDistance) {
            // Zero body case is caught earlier
            if (numBodies == 1) {
                retunr true;
            }

            Particle first = bodies[0];
            for (int i = 1; i < numBodies; i++) {
                for (int j = 1; j < numBodies; j++) {
                    if (fabs(bodies[i].position[j] - first.position[j]) >= minDistance) {
                        return false;
                    }
                }
            }
            return true;
        }

        // Create children Octree classes in the already created object
        __host__ __device__ void createChildren() {
            // Subdivide the space into 8 Octants
            for (int i = 0; i < 8; i++) {
                double newBounds[3][2];
                getSubdivisionBounds(i, newBounds); // Get bounds for current octant

                // int childNumBodies;
                // Particle* childParticles = subdivideParticles(bodies, numBodies, newBounds, &childNumBodies);

                // Octree child = Octree(subdivideParticles(bodies, numBodies, newBounds), newBounds, minDistance);
                // mass += child.mass;
                // for (int j = 0; j < 3; j++) {
                //     cm[j] += child.mass * child.cm[j];
                // }
                // children[i] = child;
            }

            if (mass > 0) {
                for (int i = 0; i < 3; i++) {
                    cm[i] /= mass;
                }
            }
        }
}