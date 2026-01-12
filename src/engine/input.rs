use std::collections::HashSet;

use winit::event::{ElementState, VirtualKeyCode};

#[derive(Clone, Debug)]
pub enum InputEvent {
    KeyPressed(VirtualKeyCode),
    KeyReleased(VirtualKeyCode),
    MouseMoved(f64, f64),
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub enum Action {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    ShutDown,
}

impl Action {
    pub fn from_key_code(keycode: VirtualKeyCode) -> Option<Self> {
        match keycode {
            VirtualKeyCode::W => Some(Action::MoveForward),
            VirtualKeyCode::S => Some(Action::MoveBackward),
            VirtualKeyCode::A => Some(Action::MoveLeft),
            VirtualKeyCode::D => Some(Action::MoveRight),
            VirtualKeyCode::Space => Some(Action::MoveUp),
            VirtualKeyCode::LShift => Some(Action::MoveDown),
            VirtualKeyCode::Escape => Some(Action::ShutDown),

            _ => None,
        }
    }
}

impl InputEvent {
    pub fn from_event_state(state: ElementState, keycode: VirtualKeyCode) -> Self {
        match state {
            ElementState::Pressed => InputEvent::KeyPressed(keycode),
            ElementState::Released => InputEvent::KeyReleased(keycode),
        }
    }

    pub fn from_mouse_motion(delta_x: f64, delta_y: f64) -> Self {
        InputEvent::MouseMoved(delta_x, delta_y)
    }
}

pub struct InputManager {
    actions: HashSet<Action>,
    mouse_delta: (f64, f64),
}

impl InputManager {
    pub fn new() -> Self {
        InputManager {
            actions: HashSet::new(),
            mouse_delta: (0.0, 0.0),
        }
    }

    pub fn is_action_active(&self, action: &Action) -> bool {
        self.actions.contains(action)
    }

    pub fn get_mouse_delta(&self) -> (f64, f64) {
        self.mouse_delta
    }

    pub fn on_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::KeyReleased(keycode) => {
                if let Some(action) = Action::from_key_code(keycode) {
                    self.actions.remove(&action);
                }
            }

            InputEvent::KeyPressed(keycode) => {
                if let Some(action) = Action::from_key_code(keycode) {
                    self.actions.insert(action);
                }
            }

            InputEvent::MouseMoved(delta_x, delta_y) => {
                self.mouse_delta = (delta_x, delta_y);
            }
        }
    }
}
