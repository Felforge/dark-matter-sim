mod config;
use config::load_config;
use nalgebra as na;

pub struct Particle {
    mass: f64,
    position: na::Vector3<f64>,
    velocity: na::Vector3<f64>,
}

// Disperse particles uniformly to a grid
pub fn disperse_to_grid() -> Option<Vec<Particle>> {
    // Load Config File
    // Question mark return None if config is None
    let config = load_config()?;

    // Create empty vector of particles
    let mut particles = Vec::new();

    // Calculate number of particles per side
    let num_per_side = (config.num_particles as f64).cbrt() as u32;

    // Calculate uniform mass per particle
    let particle_mass = config.mean_density * config.distance.powi(3) / config.num_particles;

    // Disperse particles uniformly across the grid
    for i in 0..num_per_side {
        let x = (i as f64 + 0.5) * config.distance / (num_per_side as f64);
        for j in 0..num_per_side {
            let y = (j as f64 + 0.5) * config.distance / (num_per_side as f64);
            for k in 0..num_per_side {
                let z = (k as f64 + 0.5) * config.distance / (num_per_side as f64);
                let new_particle = Particle {
                    mass: particle_mass,
                    position: na::Vector3::new(x, y, z),
                    velocity: na::Vector3::new(0.0, 0.0, 0.0)
                };
                particles.push(new_particle);
            }
        }
    }

    // Return completed vector of particles
    Some(particles)
}