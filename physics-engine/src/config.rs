use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    distance: f64,
    num_particles: u32,
    mean_density: f64,
    hubble_parameter: f64,
}

// Check if number is a perfect cube
fn is_cube(n: u32) -> bool {
    let root = (n as f64).cbrt().round();
    return (root*root*root - (n as f64)).abs() < 1e-6
}

// Load config.json into the program
pub fn load_config() -> Option<Config> {
    let file = "../../config.json";
    
    let contents = match std::fs::read_to_string(file) {
        Ok(c) => c,
        Err(e) => {
            println!("Error reading file: {}", e);
            return None;
        }
    };

    let filled_config: Config = match serde_json::from_str(&contents) {
        Ok(config) => config,
        Err(e) => {
            println!("Error parsing JSON: {}", e);
            return None;
        }
    };

    if !is_cube(filled_config.num_particles) {
        println!("num_particles must be a cube");
        return None;
    }
    Some(filled_config)
}