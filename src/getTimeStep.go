package main

/*
#cgo LDFLAGS: -L./libs -lGetTimeStep -lcudart
#include <stdlib.h>

typedef struct {
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
} Particle;

extern double getTimeStep(Particle* particles, int numParticles, double timeStepParameter, double softeningDivisor);
*/
import "C"
import "unsafe"

// GetTimeStep calls the CUDA function of the same name
func GetTimeStep(particles []Particle, timeStepParameter float64, softeningDivisor float64) float64 {
	particlesPtr := (*C.Particle)(unsafe.Pointer(&particles))
	numParticles := C.int(len(particles))
	cTimeStepParameter := C.double(timeStepParameter)
	cSofteningDivisor := C.double(softeningDivisor)
	return float64(C.getTimeStep(particlesPtr, numParticles, cTimeStepParameter, cSofteningDivisor))
}
