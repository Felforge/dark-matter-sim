package main

import "fmt"

func main() {
	config := LoadConfig()
	particles := DisperseToGrid()
	fmt.Println(particles[0].X, particles[0].Y, particles[0].Z)
	fmt.Println(particles[1].X, particles[1].Y, particles[1].Z)
	fmt.Println(particles[2].X, particles[2].Y, particles[2].Z)
	timeStep := GetTimeStep(particles, config.TimeStepParameter, config.SofteningDivisor)
	fmt.Println(timeStep)
}
