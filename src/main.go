package main

import (
	"fmt"
	"math"
	"time"
)

func computeTimeStep(particles []Particle, inpConfig Config) float64 {
	totalD := 0.0
	totalA := 0.0
	for _, p := range particles {
		totalA += p.Ax*p.Ax + p.Ay*p.Ay + p.Az*p.Az
		accumulatedD := 0.0
		for _, iP := range particles {
			if p == iP {
				continue
			}
			accumulatedD += math.Sqrt(p.X*p.X + p.Y*p.Y + p.Z*p.Z)
		}
		totalD += accumulatedD / float64(inpConfig.NumParticles-1)
	}
	totalA /= float64(inpConfig.NumParticles)
	totalD /= float64(inpConfig.NumParticles)
	return inpConfig.TimeStepParameter * math.Sqrt(totalD/totalA/inpConfig.SofteningDivisor)
}

func main() {
	config := LoadConfig()
	particles := DisperseToGrid()
	start := time.Now()
	timeStep := GetTimeStep(particles, config.TimeStepParameter, config.SofteningDivisor)
	elapsed := time.Since(start)
	fmt.Println(elapsed, timeStep)
	start = time.Now()
	timeStep = computeTimeStep(particles, config)
	elapsed = time.Since(start)
	fmt.Println(elapsed, timeStep)
}
