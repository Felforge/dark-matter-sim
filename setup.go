package main

import (
	"math"
)

// Particle structure
type Particle struct {
	Mass float32
	X    float32
	Y    float32
	Z    float32
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
	mass := config.MeanDensity * float32(math.Pow(float64(config.Distance), 3))
	mass *= float32(math.Pow(float64(config.HubbleParameter), 2))
	mass /= float32(config.NumParticles)

	// Place particles
	for i := 0.0; i < numPerSide; i++ {
		for j := 0.0; j < numPerSide; j++ {
			for k := 0.0; k < numPerSide; k++ {
				x := float32(i+0.5) * config.Distance / float32(config.NumParticles)
				y := float32(j+0.5) * config.Distance / float32(config.NumParticles)
				z := float32(k+0.5) * config.Distance / float32(config.NumParticles)
				newParticle := Particle{mass, x, y, z}
				particles = append(particles, newParticle)
			}
		}
	}

	// Return final construct
	return particles
}
