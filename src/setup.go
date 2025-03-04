package main

import (
	"math"
	"math/rand"
)

// Particle structure
// Matches CUDA c++
type Particle struct {
	Mass float64
	X    float64
	Y    float64
	Z    float64
	Vx   float64
	Vy   float64
	Vz   float64
	Ax   float64
	Ay   float64
	Az   float64
}

// Check if net accelerations are "close enough" to zero
// Put zero leniency for the default value of 1e-6
func isNearZeroAcceleration(particles []Particle, leniency float64) bool {
	// Set default value
	if leniency == 0.0 {
		leniency = 1e-6
	}

	// Iterate Particles List
	for _, p := range particles {
		// Calculate magnitude of acceleration vector
		magnitudeA := math.Sqrt(p.Ax*p.Ax + p.Ay*p.Ay + p.Az*p.Az)

		// If acceleration is greater than the maximum allowed return false
		if magnitudeA > leniency {
			return false
		}
	}

	// Return true if and only if nothing satisifies the if statement
	return true
}

// CreateGlass disperses particles in a glass configuration
func CreateGlass() []Particle {
	// Load config file
	config := LoadConfig()

	// Create particles list
	particles := []Particle{}

	// Calculate uniform mass
	mass := config.MeanDensity * config.Distance * config.Distance * config.Distance
	mass *= config.HubbleParameter * config.HubbleParameter
	mass /= float64(config.NumParticles)

	// Disperse particles randomly
	for i := 0; i < config.NumParticles; i++ {
		x := rand.Float64() * config.Distance
		y := rand.Float64() * config.Distance
		z := rand.Float64() * config.Distance
		newParticle := Particle{mass, x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
		particles = append(particles, newParticle)
	}

	// Apply initial accelerations for time step calculation
	particles = ApplyForces(particles, -1.0, config)

	// Loop until all accelerations are nearly zero
	for isNearZeroAcceleration(particles, 0.0) {
		// Compute appropriate time step
		timeStep := GetTimeStep(particles, config)
		// Time evolve using the time step
		// Mode of -1.0 is needed for antigravity
		particles = ApplyYoshida(particles, timeStep, -1.0, config)
	}

	// Return finalized particles
	return particles
}
