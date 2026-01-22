use nalgebra_glm::Vec3;

use crate::instance::{Instance, Mesh};

// Make sure res is power of 2 for best results
fn create_grid_mesh(res: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for z in 0..=res {
        for x in 0..=res {
            let u = x as f32 / res as f32;
            let v = z as f32 / res as f32;
            vertices.push(crate::instance::Vertex {
                position: [u - 0.5, 0.0, v - 0.5], // -0.5 to 0.5
                uv: [u, v],
            });
        }
    }

    for z in 0..res {
        for x in 0..res {
            let top_left = z * (res + 1) + x;
            let top_right = top_left + 1;
            let bottom_left = (z + 1) * (res + 1) + x;
            let bottom_right = bottom_left + 1;

            // First triangle (top-left, bottom-left, top-right)
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);

            // Second triangle (top-right, bottom-left, bottom-right)
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    Mesh { vertices, indices }
}

fn create_instance(pos: Vec3) -> Instance {
    let translation = nalgebra_glm::translation(&pos);
    let scale = nalgebra_glm::scaling(&Vec3::new(200.0, 1.0, 200.0));
    let model = translation * scale;
    let normal = nalgebra_glm::inverse_transpose(scale);
    Instance {
        instance_normal: normal.into(),
        instance_model: model.into(),
    }
}

pub struct Water {
    pub instances: Vec<Instance>,
    pub mesh: Mesh,
}

impl Water {
    pub fn new() -> Self {
        let mesh = create_grid_mesh(2048);

        let mut instances = Vec::new();
        instances.push(create_instance(Vec3::new(0.0, 0.0, 0.0)));

        Water { instances, mesh }
    }
}
