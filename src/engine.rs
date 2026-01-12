pub mod camera;
pub mod input;
pub mod mesh;
pub mod water;

use std::sync::{Arc, Mutex};

pub use camera::Camera;
pub use input::{Action, InputEvent, InputManager};
pub use water::Water;

pub struct Engine {
    pub input: Arc<Mutex<InputManager>>,
    pub camera: Arc<Mutex<Camera>>,
    pub water: Arc<Water>,
}

impl Engine {
    pub fn new(input: Arc<Mutex<InputManager>>, camera: Arc<Mutex<Camera>>) -> Self {
        let water = Arc::new(Water::new());

        Engine {
            camera,
            input,
            water,
        }
    }

    pub fn init(&self) {
        println!("Engine initialized");
    }

    pub fn tick(&self, delta_time: f64) {
        let mut camera = self.camera.lock().unwrap();
        let mut input = self.input.lock().unwrap();

        if input.is_action_active(&Action::ShutDown) {
            println!("Shutting down engine...");
            std::process::exit(0);
        }

        camera.tick(&mut input, delta_time);
    }
}
