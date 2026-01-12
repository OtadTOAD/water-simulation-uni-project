use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use nalgebra_glm::Vec3;

use crate::engine::mesh::{Mesh, Vertex};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct WaterInstance {
    pub instance_model: [[f32; 4]; 4],
    pub instance_normal: [[f32; 4]; 4],
    pub res: u32,
}

pub struct Water {
    pub instances: Vec<WaterInstance>,
}

// Generates plane mesh(with grid depending on resolution)
// We assume scale of 1.0 for now
fn generate_mesh(res: u32) -> Mesh {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for z in 0..res {
        for x in 0..res {
            let x_pos = (x as f32 / (res - 1) as f32) - 0.5;
            let z_pos = (z as f32 / (res - 1) as f32) - 0.5;

            let u = x as f32 / (res - 1) as f32;
            let v = z as f32 / (res - 1) as f32;

            vertices.push(Vertex {
                position: [x_pos, 0.0, z_pos],
                normal: [0.0, 1.0, 0.0],       // Plane faces up (Y+)
                tangent: [1.0, 0.0, 0.0, 1.0], // Tangent along X axis, w=handedness
                uv: [u, v],
            });
        }
    }

    for z in 0..(res - 1) {
        for x in 0..(res - 1) {
            let top_left = z * res + x;
            let top_right = top_left + 1;
            let bottom_left = (z + 1) * res + x;
            let bottom_right = bottom_left + 1;

            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);

            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    Mesh::new(vertices, indices)
}

fn generate_drawable_mesh(res: u32, pos: Vec3) -> WaterInstance {
    let translation = nalgebra_glm::translation(&pos);
    let scale = nalgebra_glm::scaling(&Vec3::new(1.0, 1.0, 1.0));
    let model = translation * scale;
    let normal = nalgebra_glm::inverse_transpose(scale);

    WaterInstance {
        instance_normal: normal.into(),
        instance_model: model.into(),
        res,
    }
}

impl Water {
    pub fn new() -> Self {
        let mesh = generate_drawable_mesh(10, Vec3::new(0.0, 0.0, 0.0));
        Water {
            instances: vec![mesh],
        }
    }

    pub fn get_draw_batches(&self) -> HashMap<u32, Vec<&WaterInstance>> {
        self.instances
            .iter()
            .map(|instance| (instance.res, vec![instance]))
            .collect()
    }

    // TODO: Cache meshes for different resolutions
    pub fn get_mesh_for_res(&self, res: u32) -> Mesh {
        generate_mesh(res)
    }
}
