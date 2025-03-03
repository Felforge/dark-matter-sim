package main

/*
#cgo LDFLAGS: -L./../libs -l:GetTimeStep.dll -lcudart
#include <stdlib.h>

typedef struct {
    double Mass;
    double X, Y, Z;
    double Vx, Vy, Vz;
    double Ax, Ay, Az;
} Particle;

__declspec(dllexport) extern double getTimeStep(Particle* particles, int numParticles, double timeStepParameter, double softeningDivisor);
*/
import "C"
import (
	"unsafe"
)

// GetTimeStep calls the CUDA function of the same name
func GetTimeStep(particles []Particle, timeStepParameter float64, softeningDivisor float64) float64 {
	particlesPtr := (*C.Particle)(unsafe.Pointer(&particles[0]))
	numParticles := C.int(len(particles))
	cTimeStepParameter := C.double(timeStepParameter)
	cSofteningDivisor := C.double(softeningDivisor)
	ts := C.getTimeStep(particlesPtr, numParticles, cTimeStepParameter, cSofteningDivisor)
	return float64(ts)
}
