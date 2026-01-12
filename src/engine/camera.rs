use crate::engine::input::InputManager;
use nalgebra_glm as glm;

const MOVE_SPEED: f32 = 5.0;
const ROTATE_SPEED: f32 = 0.005;
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

pub struct Camera {
    pub position: glm::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,

    proj: glm::Mat4,
    view: glm::Mat4,
    pub is_dirty: bool,
}

impl Camera {
    pub fn new(position: glm::Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            fov: 70.0_f32.to_radians(),
            proj: glm::Mat4::identity(),
            view: glm::Mat4::identity(),
            is_dirty: true,
        }
    }

    pub fn forward(&self) -> glm::Vec3 {
        glm::vec3(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    pub fn right(&self) -> glm::Vec3 {
        glm::vec3(
            (self.yaw + std::f32::consts::FRAC_PI_2).cos(),
            0.0,
            (self.yaw + std::f32::consts::FRAC_PI_2).sin(),
        )
    }

    /* Don't need this for now
    pub fn up(&self) -> glm::Vec3 {
        glm::cross(&self.right(), &self.forward())
    }*/

    pub fn move_forward(&mut self, distance: f32) {
        self.position += self.forward() * distance;
        self.is_dirty = true;
    }

    pub fn move_backward(&mut self, distance: f32) {
        self.move_forward(-distance);
    }

    pub fn move_right(&mut self, distance: f32) {
        self.position += self.right() * distance;
        self.is_dirty = true;
    }

    pub fn move_left(&mut self, distance: f32) {
        self.move_right(-distance);
    }

    pub fn move_up(&mut self, distance: f32) {
        self.position.y -= distance; // Vulkan Y is down
        self.is_dirty = true;
    }

    pub fn move_down(&mut self, distance: f32) {
        self.move_up(-distance);
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.is_dirty = true;
    }

    pub fn update_matrices(&mut self, aspect_ratio: f32) {
        if !self.is_dirty {
            return;
        }

        // Perspective projection for Vulkan (reverse Z for better depth precision)
        self.proj = glm::perspective_fov_rh_zo(self.fov, aspect_ratio, 1.0, 0.1, 1000.0);

        // View matrix: look from position in the direction we're facing
        let target = self.position + self.forward();
        self.view = glm::look_at_rh(&self.position, &target, &glm::Vec3::y());

        self.is_dirty = false;
    }

    pub fn tick(&mut self, input: &mut InputManager, delta_time: f64) {
        let dt = delta_time as f32;

        if input.is_action_active(&super::input::Action::MoveForward) {
            self.move_forward(MOVE_SPEED * dt);
        }
        if input.is_action_active(&super::input::Action::MoveBackward) {
            self.move_backward(MOVE_SPEED * dt);
        }
        if input.is_action_active(&super::input::Action::MoveLeft) {
            self.move_left(MOVE_SPEED * dt);
        }
        if input.is_action_active(&super::input::Action::MoveRight) {
            self.move_right(MOVE_SPEED * dt);
        }
        if input.is_action_active(&super::input::Action::MoveUp) {
            self.move_up(MOVE_SPEED * dt);
        }
        if input.is_action_active(&super::input::Action::MoveDown) {
            self.move_down(MOVE_SPEED * dt);
        }

        let (delta_x, delta_y) = input.take_mouse_delta();
        if delta_x != 0.0 || delta_y != 0.0 {
            self.rotate(delta_x as f32 * ROTATE_SPEED, delta_y as f32 * ROTATE_SPEED);
        }
    }
}

impl Camera {
    pub fn view_matrix_raw(&self) -> [[f32; 4]; 4] {
        //glm::transpose(&self.view).into()
        self.view.into()
    }

    pub fn projection_matrix_raw(&self) -> [[f32; 4]; 4] {
        //glm::transpose(&self.proj).into()
        self.proj.into()
    }
}
