package main

import "fmt"

func main() {
	config := LoadConfig()
	particles := DisperseToGrid()
	timeStep := GetTimeStep(particles, config.TimeStepParameter, config.SofteningDivisor)
	fmt.Println(timeStep)
}