mod config;
use config::load_config;

fn main() {
    match load_config() {
        Some(config) => {
            println!("{:#?}", config);
        },
        None => {
            println!("Failed to load config");
        }
    }
}