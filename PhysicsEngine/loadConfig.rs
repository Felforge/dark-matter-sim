// USE "cargo new" COMMAND TO CREATE ACTUAL PROJECT

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Config {
    distance: f64
    num_particles: u32
    mean_density: f64
    hubbe_parameter: f64
}

// Check if number is a perfect cube
fn is_cube(n: u32) -> bool {
    let root = (n as f64).cbrt().round();
    return (root**3 - (n as f64)).abs() < 1e-6
}

// Load config.json into the program
fn load_config(filename: str = "config.json") -> Config {
    let file = filename;
    let config: Config = serde_json::from_str(file).expect("JSON was not well-formatted");
    if not is_cube(config.num_particles) {
        println!("num_particles must be a cube")
        return None
    }
    return config
}