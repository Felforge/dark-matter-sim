package main

/*
#cgo LDFLAGS: -L./build -lbarneshut -lcudart
#include <stdlib.h>

// Define the Particle struct in C
typedef struct {
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
} Particle;

// Declare external functions
extern void* createBarnesHut(int numBodies, Particle* particles, double bounds[3][2]);
extern void destroyBarnesHut(void* tree);
extern void computeForces(void* tree, double theta, double mode);
extern void copyParticlesToHost(void* tree, Particle* hostParticles, int numBodies);
*/
import "C"
import (
	"unsafe"
)

// Fetch updated particles from CUDA and return as Go slice
func getParticles(tree unsafe.Pointer, numBodies int) []Particle {
	hostParticles := make([]C.Particle, numBodies)

	// Copy CUDA particles to CPU (host memory)
	C.copyParticlesToHost(tree, (*C.Particle)(unsafe.Pointer(&hostParticles[0])), C.int(numBodies))

	// Convert C array to Go slice
	goParticles := make([]Particle, numBodies)
	for i := 0; i < numBodies; i++ {
		goParticles[i] = Particle{
			Mass: float64(hostParticles[i].mass),
			X:    float64(hostParticles[i].x), Y: float64(hostParticles[i].y), Z: float64(hostParticles[i].z),
			Vx: float64(hostParticles[i].vx), Vy: float64(hostParticles[i].vy), Vz: float64(hostParticles[i].vz),
			Ax: float64(hostParticles[i].ax), Ay: float64(hostParticles[i].ay), Az: float64(hostParticles[i].az),
		}
	}
	return goParticles
}

// Mode is 1 for normal, -1 for inverse
// Enter 0 for theta or mode to use fault value
func computeForces(distance float64, particles []Particle, theta float64, mode float64) []Particle {
	if theta == 0 {
		theta = 0.5
	}
	if mode == 0 {
		mode = 1.0
	}

	particlesPtr := (*C.Particle)(unsafe.Pointer(&particles))

	var bounds [3][2]C.double
	for i := 0; i < 3; i++ {
		bounds[i] = [2]C.double{0.0, C.double(distance)}
	}
	boundsPtr := (*[2]C.double)(unsafe.Pointer(&bounds[0]))

	tree := C.createBarnesHut(C.int(len(particles)), particlesPtr, boundsPtr)
	defer C.destroyBarnesHut(tree)

	C.computeForces(tree, C.double(theta), C.double(mode))

	updatedParticles := getParticles(tree, len(particles))

	return updatedParticles
}
