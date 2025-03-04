package main

import "fmt"

func main() {
	particles := CreateGlass()
	p := particles[1000]
	fmt.Println(p.X, p.Y, p.Z)
}
