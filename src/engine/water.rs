use crate::engine::mesh::{Mesh, Vertex};

pub struct Water {
    pub mesh: Mesh,
}

// Generates plane mesh(with grid depending on resolution)
// We assume scale of 1.0 for now
fn generate_mesh(resolution: u32) -> Mesh {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for z in 0..resolution {
        for x in 0..resolution {
            let x_pos = (x as f32 / (resolution - 1) as f32) - 0.5;
            let z_pos = (z as f32 / (resolution - 1) as f32) - 0.5;

            let u = x as f32 / (resolution - 1) as f32;
            let v = z as f32 / (resolution - 1) as f32;

            vertices.push(Vertex {
                position: [x_pos, 0.0, z_pos],
                normal: [0.0, 1.0, 0.0],       // Plane faces up (Y+)
                tangent: [1.0, 0.0, 0.0, 1.0], // Tangent along X axis, w=handedness
                uv: [u, v],
            });
        }
    }

    for z in 0..(resolution - 1) {
        for x in 0..(resolution - 1) {
            let top_left = z * resolution + x;
            let top_right = top_left + 1;
            let bottom_left = (z + 1) * resolution + x;
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

impl Water {
    pub fn new() -> Self {
        let mesh = generate_mesh(10);
        Water { mesh }
    }
}
