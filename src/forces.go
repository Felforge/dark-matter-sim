package main

/*
#cgo LDFLAGS: -L./../libs -l:Forces.dll -lcudart
#include <stdlib.h>

// Define the Particle struct in C
typedef struct {
    double mass;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
} Particle;

// Declare external functions
// Both update the inputted particles parameter externally
__declspec(dllexport) void applyYoshida(int numParticles, Particle* particles, double distance, double dt, double theta, double mode);
__declspec(dllexport) void applyForces(int numParticles, Particle* particles, double distance, double theta, double mode);
*/
import "C"
import "unsafe"

// Mode is 1.0 for normal, -1.0 for inverse
// Theta is for the Barnes-Hut Algorithm and is in the config
// Enter 0 for theta or mode to use the default value
// Return may not be needed need to test
func ApplyYoshida(particles []Particle, dt float64, mode float64, config Config) []Particle {
	// Set default value
	if mode == 0.0 {
		mode = 1.0
	}

	// Get pointer for particles
	particlesPtr := (*C.Particle)(unsafe.Pointer(&particles[0]))

	// Get C Variables
	cNumParticles := C.int(len(particles))
	cDistance := C.double(config.Distance)
	cDt := C.double(dt)
	cTheta := C.double(config.Theta)
	cMode := C.double(mode)

	// Update particles
	C.applyYoshida(cNumParticles, particlesPtr, cDistance, cDt, cTheta, cMode)

	// Return particles that is now updated
	return particles
}

// Mode is 1.0 for normal, -1.0 for inverse
// Theta is for the Barnes-Hut Algorithm
// Enter 0 for theta or mode to use the default value
func ApplyForces(particles []Particle, mode float64, config Config) []Particle {
	// Set default values
	if mode == 0.0 {
		mode = 1.0
	}

	// Get pointer for particles
	particlesPtr := (*C.Particle)(unsafe.Pointer(&particles[0]))

	// Get C Variables
	cNumParticles := C.int(len(particles))
	cDistance := C.double(config.Distance)
	cTheta := C.double(config.Theta)
	cMode := C.double(mode)

	// Update particles
	C.applyForces(cNumParticles, particlesPtr, cDistance, cTheta, cMode)

	// Return particles that is now updated
	return particles
}
