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

// DisperseToGrid disperses particles uniformly to a grid
func DisperseToGrid() []Particle {
	// Load config file
	config := LoadConfig()

	// Create particles list
	particles := []Particle{}

	// Calculate spacing
	numPerSide := math.Cbrt(float64(config.NumParticles))

	// Calculate uniform mass
	mass := config.MeanDensity * (math.Pow(float64(config.Distance), 3))
	mass *= math.Pow(float64(config.HubbleParameter), 2)
	mass /= float64(config.NumParticles)

	// Place particles
	for i := 0.0; i < numPerSide; i++ {
		for j := 0.0; j < numPerSide; j++ {
			for k := 0.0; k < numPerSide; k++ {
				x := (i + 0.5) * config.Distance / float64(config.NumParticles)
				y := (j + 0.5) * config.Distance / float64(config.NumParticles)
				z := (k + 0.5) * config.Distance / float64(config.NumParticles)
				newParticle := Particle{mass, x, y, z, 0, 0, 0, 0, 0, 0}
				particles = append(particles, newParticle)
			}
		}
	}

	// Return final construct
	return particles
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
}

// Use Yoshida integration to implement antigravity
