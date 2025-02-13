use crate::config::Config;

#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub struct Particle {
    pub mass: f64,
    pub x: f64, pub y: f64, pub z: f64,
    pub vx: f64, pub vy: f64, pub vz: f64,
    pub ax: f64, pub ay: f64, pub az: f64,
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
                    x: x, y: y, z: z,
                    vx: 0.0, vy: 0.0, vz: 0.0,
                    ax: 0.0, ay: 0.0, az: 0.0,
                };
                particles.push(new_particle);
            }
        }
    }

    // Return completed vector of particles
    Some(particles)
}

// Disperse particles in glass configuration