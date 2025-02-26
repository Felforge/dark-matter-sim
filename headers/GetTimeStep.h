#ifndef TIME_STEP_H
#define TIME_STEP_H

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
double getTimeSetp(Particle* particles, int numParticles, double timeStepParameter, double softeningDivisor);

#ifdef __cplusplus
}
#endif

#endif // BARNES_HUT_H