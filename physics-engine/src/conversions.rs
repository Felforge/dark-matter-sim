use physical_constants;

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
    let pi: f64 = std::f64::consts::PI;
    let gravitational_constant = physical_constants::NEWTONIAN_CONSTANT_OF_GRAVITATION as f64;
    let au: f64 = 149597870700.0;
    let multiple: f64 = gravitational_constant * 31536000.0 * 31536000.0 * m;
    let divisor: f64 = 4.0 * pi * pi * au * au * au;
    return divisor / multiple
}

// Convert mean density into base units
fn convert_density(density: f64, h: f64) -> f64 {
    let mpc_conversion: f64 = convert_distance(1.0);
    let mass_conversion: f64 = convert_mass(1.0);
    return density * mass_conversion * h.powi(2) / mpc_conversion.powi(3)
}