package main

import "math"

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
	numPerSide := float32(math.Cbrt(float64(config.NumParticles)))
	spacing := config.Distance / (numPerSide - 1)

	// Calculate uniform mass
	mass := (config.MeanDensity * float32(math.Pow(float64(config.Distance), 3))) / float32(config.NumParticles)

	// Place particles
	var x, y, z float32
	for x = 0; x < config.Distance; x += spacing {
		for y = 0; y < config.Distance; y += spacing {
			for z = 0; z < config.Distance; z += spacing {
				newParticle := Particle{mass, x, y, z}
				particles = append(particles, newParticle)
			}
		}
	}

	// Return final construct
	return particles
}
