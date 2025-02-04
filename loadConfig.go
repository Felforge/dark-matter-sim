package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config structure
type Config struct {
	Distance     float32 `json:"distanceMegaparsec"`
	NumParticles int     `json:"numParticles"`
	MeanDensity  float32 `json:"meanDensity"`
}

// ConvertDistance converts from megaparsec to meters
func ConvertDistance(distanceSolar float32) float32 {
	conversion := float32(9.69394202136e22 / math.Pi)
	return distanceSolar * conversion
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

	// Convert Distance to Base Units
	config.Distance = ConvertDistance(config.Distance)

	// Return Final
	return *config
}
