use crate::config::Config;
use crate::setup::Particle;
use rustacuda::prelude::*;
use rustacuda::function::*;
use rustacuda::launch;
use rustacuda::stream::Stream;
use std::ffi::CString;

pub fn apply_zeldovich_cuda(config: Config, particles: &mut Vec<Particle>, growth_factor: f64, growth_rate: f64) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    rustacuda::init(CudaFlags::empty())?;

    // Select GPU device and create context
    let device: Device = Device::get_device(0)?;
    #[allow(unused_variables)]
    let context: Context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Convert particle positions to density field
    let num_per_side: usize = (config.num_particles as f64).cbrt() as usize;
    let mut density_field: Vec<f32> = vec![0.0; num_per_side * num_per_side * num_per_side];

    for particle in particles.iter() {
        let i = ((particle.position.x / config.distance) * num_per_side as f64) as usize;
        let j = ((particle.position.y / config.distance) * num_per_side as f64) as usize;
        let k = ((particle.position.z / config.distance) * num_per_side as f64) as usize;

        // Clamp indices to prevent out-of-bounds errors
        let i = i.min(num_per_side - 1);
        let j = j.min(num_per_side - 1);
        let k = k.min(num_per_side - 1);

        density_field[i * num_per_side * num_per_side + j * num_per_side + k] += 1.0;
    }

     // Convert to density contrast
    for i in 0..density_field.len() {
        density_field[i] = (density_field[i] / config.mean_density as f32) - 1.0;
    }

    // Transfer data to GPU
    let mut dev_density: DeviceBuffer<f32> = DeviceBuffer::from_slice(&density_field)?;
    let mut dev_displacement_x: DeviceBuffer<f64> = DeviceBuffer::from_slice(&vec![0.0; density_field.len()])?;
    let mut dev_displacement_y: DeviceBuffer<f64> = DeviceBuffer::from_slice(&vec![0.0; density_field.len()])?;
    let mut dev_displacement_z: DeviceBuffer<f64> = DeviceBuffer::from_slice(&vec![0.0; density_field.len()])?;

    // Load CUDA kernel
    let module_data = std::fs::read_to_string("src/zeldovich.ptx")?;
    let module = Module::load_from_string(&CString::new(module_data)?)?;
    let func: Function<'_> = module.get_function(CString::new("compute_displacement").unwrap().as_c_str())?;

    // Launch CUDA kernel
    let stream: Stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let num_threads: usize = 256;
    let num_blocks: usize = (num_per_side.pow(3) + num_threads - 1) / num_threads;

    unsafe {
        launch!(func<<<num_blocks as u32, num_threads as u32, 0, stream>>>(
            dev_density.as_device_ptr(),
            dev_displacement_x.as_device_ptr(),
            dev_displacement_y.as_device_ptr(),
            dev_displacement_z.as_device_ptr(),
            num_per_side as i32
        ))?;
    }

    // Ensure kernel execution completes before copying results
    stream.synchronize()?;

    // Copy results back from GPU
    let mut displacement_x: Vec<f64> = vec![0.0; density_field.len()];
    let mut displacement_y: Vec<f64> = vec![0.0; density_field.len()];
    let mut displacement_z: Vec<f64> = vec![0.0; density_field.len()];

    dev_displacement_x.copy_to(&mut displacement_x)?;
    dev_displacement_y.copy_to(&mut displacement_y)?;
    dev_displacement_z.copy_to(&mut displacement_z)?;

    // Apply Zel’dovich Approximation to Particles
    for particle in particles.iter_mut() {
        let i = ((particle.position.x / config.distance) * num_per_side as f64).min(num_per_side as f64 - 1.0) as usize;
        let j = ((particle.position.y / config.distance) * num_per_side as f64).min(num_per_side as f64 - 1.0) as usize;
        let k = ((particle.position.z / config.distance) * num_per_side as f64).min(num_per_side as f64 - 1.0) as usize;

        let i = i.min(num_per_side - 1);
        let j = j.min(num_per_side - 1);
        let k = k.min(num_per_side - 1);

        let displacement_index = i * num_per_side * num_per_side + j * num_per_side + k;

        let dx: f64 = growth_factor * displacement_x[displacement_index];
        let dy: f64 = growth_factor * displacement_y[displacement_index];
        let dz: f64 = growth_factor * displacement_z[displacement_index];

        particle.position.x += dx;
        particle.position.y += dy;
        particle.position.z += dz;

        particle.velocity.x = growth_rate * dx;
        particle.velocity.y = growth_rate * dy;
        particle.velocity.z = growth_rate * dz;
    }

    Ok(())
}
