package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config structure
type Config struct {
	Distance          float64 `json:"distanceMegaparsec"`
	NumParticles      int     `json:"numParticles"`
	MeanDensity       float64 `json:"meanDensity"`
	HubbleParameter   float64 `json:"hubbleParameter"`
	Theta             float64 `json:"barnesHutTheta"`
	TimeStepParameter float64 `json:"timeStepParameter"`
	SofteningDivisor  float64 `json:"softeningDivisor"`
}

// convertDistance converts from megaparsec to meters
func convertDistance(distanceSolar float64) float64 {
	conversion := 9.69394202136e22 / math.Pi
	return distanceSolar * conversion
}

// Needs to be tested
// Convert Mass from solar mass units to kilograms
func convertMass(solarMass float64) float64 {
	G := 6.67428e-11
	pi := math.Pi
	au := 149597870700.0
	multiple := G * solarMass * math.Pow(31536000.0, 2)
	divisor := 4.0 * math.Pow(pi, 2.0) * math.Pow(au, 3.0)
	return multiple / divisor
}

// Convert Mean Density to base units
func convertDensity(solarDensity float64, h float64) float64 {
	mpc := convertDistance(1.0)
	mass := convertMass(1.0)
	return solarDensity * mass * math.Pow(h, 2.0) / math.Pow(mpc, 3.0)
}

// convertHubbleParameter converts the Hubble Parameter into base units
func convertHubbleParameter(hubbleParameter float64) float64 {
	conversionMPC := convertDistance(1)
	return 1000 * hubbleParameter / conversionMPC
}

// LoadConfig loads config.json
func LoadConfig() Config {
	// Load initial config
	config := &Config{}

	// Open Config File
	file, err := os.Open("../config.json")
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

	// Convert to base units
	config.Distance = convertDistance(config.Distance)
	config.MeanDensity = convertDensity(config.MeanDensity, config.HubbleParameter)
	config.HubbleParameter = convertHubbleParameter(config.HubbleParameter)

	// Return Final
	return *config
}
