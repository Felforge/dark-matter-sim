package main

import (
	"fmt"
)

// DisperseToGrid disperses particles uniformly to a grid
func DisperseToGrid() {
	config := LoadConfig()
	fmt.Println(config.Distance, config.MeanDensity, config.NumParticles)
}

func main() {
	DisperseToGrid()
}
