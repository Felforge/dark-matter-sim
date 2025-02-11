use crate::config::Config;
use nalgebra as na;

#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub struct Particle {
    pub mass: f64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}

// Disperse particles uniformly to a grid
pub fn disperse_to_grid(config: Config) -> Option<Vec<Particle>> {
    // Load Config File
    // Question mark return None if config is None
    //let config: crate::config::Config = load_config()?;

    // Create empty vector of particles
    let mut particles: Vec<Particle> = Vec::new();

    // Calculate number of particles per side
    let num_per_side: u32 = (config.num_particles as f64).cbrt() as u32;

    // Calculate uniform mass per particle
    let particle_mass: f64 = config.mean_density * config.distance.powi(3) / (config.num_particles as f64);

    // Disperse particles uniformly across the grid
    for i in 0..num_per_side {
        let x: f64 = (i as f64 + 0.5) * config.distance / (num_per_side as f64);
        for j in 0..num_per_side {
            let y = (j as f64 + 0.5) * config.distance / (num_per_side as f64);
            for k in 0..num_per_side {
                let z = (k as f64 + 0.5) * config.distance / (num_per_side as f64);
                let new_particle: Particle = Particle {
                    mass: particle_mass,
                    position: [x, y, z],
                    velocity: [0.0, 0.0, 0.0]
                };
                particles.push(new_particle);
            }
        }
    }

    // Return completed vector of particles
    Some(particles)
}

// Disperse particles in glass configuration