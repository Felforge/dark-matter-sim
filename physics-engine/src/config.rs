use serde::Deserialize;


#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub distance: f64,
    pub num_particles: u32,
    pub mean_density: f64,
    pub hubble_parameter: f64,
}

// Check if number is a perfect cube
fn is_cube(n: u32) -> bool {
    let root: f64 = (n as f64).cbrt().round();
    return (root.powi(3) - (n as f64)).abs() < 1e-6
}

// Load config.json into the program
pub fn load_config() -> Option<Config> {
    let file: &str = "../config.json";
    
    // Read JSON Contents
    let contents: String = match std::fs::read_to_string(file) {
        Ok(c) => c,
        Err(e) => {
            println!("Error reading file: {}", e);
            return None;
        }
    };

    // Match JSON contents to struct
    let filled_config: Config = match serde_json::from_str(&contents) {
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
    //filled_config.distance = convert_distance(filled_config.distance);
    //filled_config.mean_density = convert_density(filled_config.mean_density, filled_config.hubble_parameter);
    // Hubble parameter doesn't seem to need to be converted

    // Return Config
    // Some is needed as there is an Optional Return
    Some(filled_config)
}