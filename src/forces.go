package main

/*
#cgo LDFLAGS: -L./../libs -l:Forces.dll -lcudart_static
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
// Theta is for the Barnes-Hut Algorithm
// Enter 0 for theta or mode to use the default value
func ApplyYoshida(particles []Particle, distance float64, dt float64, theta float64, mode float64) []Particle {
	// Set default values
	if theta == 0.0 {
		theta = 0.5
	}
	if mode == 0.0 {
		mode = 1.0
	}

	// Get pointer for particles
	particlesPtr := (*C.Particle)(unsafe.Pointer(&particles[0]))

	// Get C Variables
	cNumParticles := C.int(len(particles))
	cDistance := C.double(distance)
	cDt := C.double(dt)
	cTheta := C.double(theta)
	cMode := C.double(mode)

	// Update particles
	C.applyYoshida(cNumParticles, particlesPtr, cDistance, cDt, cTheta, cMode)

	// Return particles that is now updated
	return particles
}

// Mode is 1.0 for normal, -1.0 for inverse
// Theta is for the Barnes-Hut Algorithm
// Enter 0 for theta or mode to use the default value
func ApplyForces(particles []Particle, distance float64, theta float64, mode float64) []Particle {
	// Set default values
	if theta == 0.0 {
		theta = 0.5
	}
	if mode == 0.0 {
		mode = 1.0
	}

	// Get pointer for particles
	particlesPtr := (*C.Particle)(unsafe.Pointer(&particles[0]))

	// Get C Variables
	cNumParticles := C.int(len(particles))
	cDistance := C.double(distance)
	cTheta := C.double(theta)
	cMode := C.double(mode)

	// Update particles
	C.applyForces(cNumParticles, particlesPtr, cDistance, cTheta, cMode)

	// Return particles that is now updated
	return particles
}
