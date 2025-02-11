mod config;
mod setup;
mod zeldovich;
mod barnes_hut;
use config::load_config;
use config::Config;
use setup::disperse_to_grid;
use zeldovich::apply_zeldovich_cuda;
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()>{
    let config: Config = match load_config() {
        Some(p) => p,
        None => Config { distance:0.0, num_particles: 0, mean_density: 0.0, hubble_parameter: 0.0 },
    };
    let mut particles: Vec<setup::Particle> = match disperse_to_grid(config.clone()) {
        Some(p) => p,
        None => Vec::new(),
    };
    apply_zeldovich_cuda(config, &mut particles, 1.0, 0.1).unwrap();

    // Print all particles onto document
    let mut file: File = File::create("src/particles.txt")?;
    for (i, particle) in particles.iter().enumerate() {
        writeln!(file, "Particle {}: {:?}", i, particle)?;
    }

    Ok(())
}
