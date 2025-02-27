package main

/*
#cgo LDFLAGS: -L/libs -lBarnesHut -lcudart
#include <stdlib.h>

// Define the Particle struct in C
typedef struct {
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
} Particle;

// Define the BarnesHut struct in C
typedef struct BarnesHut BarnesHut;

// Declare external functions
extern BarnesHut* createBarnesHut(int numBodies, Particle* bodies, double* bounds);
extern void destroyBarnesHut(BarnesHut* tree);
extern void computeBarnesHutForces(BarnesHut* tree, double theta, double mode);
extern void copyParticlesToHost(BarnesHut* tree, int numBodies, Particle* hBodies);
*/
import "C"
import (
	"unsafe"
)

// Mode is 1 for normal, -1 for inverse
// Enter 0 for theta or mode to use fault value
// Updates all particle accelerations
func computeForces(distance float64, particles []Particle, theta float64, mode float64) []Particle {
	if theta == 0 {
		theta = 0.5
	}
	if mode == 0 {
		mode = 1.0
	}

	// Create pointer to list of particles
	particlesPtr := (*C.Particle)(unsafe.Pointer(&particles))

	// Create pointer to bounds
	// Bounds are the same across 3 dimensions for the whole area
	var bounds [3][2]C.double
	for i := 0; i < 3; i++ {
		bounds[i][0] = 0.0
		bounds[i][1] = C.double(distance)
	}
	boundsPtr := (*C.double)(unsafe.Pointer(&bounds[0][0]))

	// C-compatibel list of bodies
	numBodies := len(particles)
	cNumBodies := C.int(numBodies)

	// Create Barnes Hut Tree
	tree := C.createBarnesHut(cNumBodies, particlesPtr, boundsPtr)

	// Compute forces
	cTheta := C.double(theta)
	cMode := C.double(mode)
	C.computeBarnesHutForces(tree, cTheta, cMode)

	// Retrieve updated particles
	hParticles := make([]C.Particle, numBodies)
	hParticlesPtr := (*C.Particle)(unsafe.Pointer(&hParticles))
	C.copyParticlesToHost(tree, cNumBodies, hParticlesPtr)

	// Convert from C Particle to Go Particle
	goParticles := make([]Particle, numBodies)
	for i := 0; i < numBodies; i++ {
		goParticles[i] = Particle{
			Mass: float64(hParticles[i].mass),
			X:    float64(hParticles[i].x), Y: float64(hParticles[i].y), Z: float64(hParticles[i].z),
			Vx: float64(hParticles[i].vx), Vy: float64(hParticles[i].vy), Vz: float64(hParticles[i].vz),
			Ax: float64(hParticles[i].ax), Ay: float64(hParticles[i].ay), Az: float64(hParticles[i].az),
		}
	}
	return goParticles
}
