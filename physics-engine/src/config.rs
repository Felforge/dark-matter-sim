use physical_constants;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    distance: f64,
    num_particles: u32,
    mean_density: f64,
    hubble_parameter: f64,
}

// Convert distance into base units
fn convert_distance(distance: f64) -> f64 {
    let pi = std::f64::consts::PI;
    return distance * 9.69394202136e22 / pi
}

// Convert hubble parameter into base units
fn convert_hubble(h: f64) -> f64 {
    let mpc_conversion = convert_distance(1.0);
    return h * 1000.0 / mpc_conversion
}

// Convert mass in solar units to kilograms
fn convert_mass(m: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let G = physical_constants::NEWTONIAN_CONSTANT_OF_GRAVITATION as f64;
    let AU = 149597870700.0;
    let multiple = G * 31536000.0 * 31536000.0 * m;
    let divisor = 4.0 * pi * pi * AU * AU * AU;
    return divisor / multiple
}

// Convert mean density into base units
fn convert_density(density: f64, h: f64) -> f64 {
    let mpc_conversion = convert_distance(1.0);
    let mass_conversion = convert_mass(1.0);
    return density * mass_conversion * h.powi(2) / mpc_conversion.powi(3)
}

// Check if number is a perfect cube
fn is_cube(n: u32) -> bool {
    let root = (n as f64).cbrt().round();
    return (root.powi(3) - (n as f64)).abs() < 1e-6
}

// Load config.json into the program
pub fn load_config() -> Option<Config> {
    let file = "../config.json";
    
    // Read JSON Contents
    let contents = match std::fs::read_to_string(file) {
        Ok(c) => c,
        Err(e) => {
            println!("Error reading file: {}", e);
            return None;
        }
    };

    // Match JSON contents to struct
    let mut filled_config: Config = match serde_json::from_str(&contents) {
        Ok(config) => config,
        Err(e) => {
            println!("Error parsing JSON: {}", e);
            return None;
        }
    };

    // Make sure num of particles is a perfect cube as needed
    if !is_cube(filled_config.num_particles) {
        println!("Error in JSON contents: num_particles must be a perfect cube");
        return None;
    }

    // Convert distance and hubble parameter to base units
    filled_config.distance = convert_distance(filled_config.distance);
    // Non-unit-converted hubble parameter must be used for density
    filled_config.mean_density = convert_density(filled_config.mean_density, filled_config.hubble_parameter);
    filled_config.hubble_parameter = convert_hubble(filled_config.hubble_parameter);

    // Return Config
    // Some is needed as there is an Optional Return
    Some(filled_config)
}