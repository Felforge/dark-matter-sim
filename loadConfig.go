package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config structure
type Config struct {
	Distance        float32 `json:"distanceMegaparsec"`
	NumParticles    int     `json:"numParticles"`
	MeanDensity     float32 `json:"meanDensity"`
	HubbleParameter float32 `json:"hubbleParameter"`
}

// convertDistance converts from megaparsec to meters
func convertDistance(distanceSolar float32) float32 {
	conversion := float32(9.69394202136e22 / math.Pi)
	return distanceSolar * conversion
}

// convertHubbleParameter converts the Hubble Parameter into base units
func convertHubbleParameter(hubbleParameter float32) float32 {
	conversionMPC := convertDistance(1)
	return 1000 * hubbleParameter / conversionMPC
}

// isCube checks if a given number is a cube
func isCube(n int) bool {
	cubeRoot := math.Cbrt(float64(n))
	return math.Round(cubeRoot) == cubeRoot
}

// LoadConfig loads config.json
func LoadConfig() Config {
	// Load initial config
	config := &Config{}

	// Open Config File
	file, err := os.Open("config.json")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return *config
	}
	defer file.Close()

	// Unpack Config
	decoder := json.NewDecoder(file)
	err = decoder.Decode(config)
	if err != nil {
		fmt.Println("Error decoding JSON:", err)
		return *config
	}

	// Check if NumParticles is a cube
	if isCube(config.NumParticles) == false {
		fmt.Println("Error loading JSON: numParticles must be a cube")
	}

	// Convert Items to Base Units
	config.Distance = convertDistance(config.Distance)
	config.HubbleParameter = convertHubbleParameter(config.HubbleParameter)

	// Return Final
	return *config
}
