#ifndef BARNES_HUT_H
#define BARNES_HUT_H

#ifdef __cplusplus
extern "C" {
#endif

// Define the Particle struct so Go can use it
typedef struct {
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
} Particle;

// C-compatible wrapper functions
void* createBarnesHut(int numBodies, Particle* particles, double bounds[3][2]);
void computeForces(void* tree, double theta, double mode);
void destroyBarnesHut(void* tree);
void copyParticlesToHost(void* tree, Particle* hostParticles, int numBodies);

#ifdef __cplusplus
}
#endif

#endif // BARNES_HUT_H