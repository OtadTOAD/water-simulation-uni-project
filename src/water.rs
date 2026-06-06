use nalgebra_glm::Vec3;

use crate::instance::{Instance, Mesh};

const TILE_RES: u32 = 64;
const BASE_TILE_SIZE: f32 = 16.0;
const LODS: u32 = 9;
const TILES_PER_SIDE: i32 = 6;
const MORPH_REGION: f32 = 0.75;

// Make sure res is power of 2 for best results
fn create_tile_mesh(res: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for z in 0..=res {
        for x in 0..=res {
            let u = x as f32 / res as f32;
            let v = z as f32 / res as f32;
            vertices.push(crate::instance::Vertex {
                position: [u, 0.0, v], // unit tile, placed and scaled per instance
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

            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);

            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    Mesh { vertices, indices }
}

fn tile_instance(origin_x: f32, origin_z: f32, size: f32, level: u32, morph_start: f32, morph_end: f32) -> Instance {
    let translation = nalgebra_glm::translation(&Vec3::new(origin_x, 0.0, origin_z));
    let scale = nalgebra_glm::scaling(&Vec3::new(size, 1.0, size));
    Instance {
        instance_normal: nalgebra_glm::Mat4::identity().into(),
        instance_model: (translation * scale).into(),
        lod_morph: [morph_start, morph_end, size / TILE_RES as f32, level as f32],
    }
}

pub struct Water {
    pub instances: Vec<Instance>,
    pub mesh: Mesh,
    last_snap: (i64, i64),
}

impl Water {
    pub fn new() -> Self {
        let mut water = Water {
            instances: Vec::new(),
            mesh: create_tile_mesh(TILE_RES),
            last_snap: (i64::MIN, i64::MIN),
        };
        water.update(Vec3::new(0.0, 0.0, 0.0));
        water
    }

    pub fn update(&mut self, camera_pos: Vec3) -> bool {
        let snap = (
            (camera_pos.x / BASE_TILE_SIZE).floor() as i64,
            (camera_pos.z / BASE_TILE_SIZE).floor() as i64,
        );
        if snap == self.last_snap {
            return false;
        }
        self.last_snap = snap;

        self.instances.clear();
        let half = TILES_PER_SIDE / 2;

        for l in 0..LODS {
            let size = BASE_TILE_SIZE * (1u32 << l) as f32;
            let morph_end = (half - 1) as f32 * size;
            let morph_start = morph_end * MORPH_REGION;

            let center_x = (camera_pos.x / size).floor() * size;
            let center_z = (camera_pos.z / size).floor() * size;

            let finer = size * 0.5;
            let finer_half = half as f32 * finer;
            let finer_cx = (camera_pos.x / finer).floor() * finer;
            let finer_cz = (camera_pos.z / finer).floor() * finer;

            for j in -half..half {
                for i in -half..half {
                    let ox = center_x + i as f32 * size;
                    let oz = center_z + j as f32 * size;

                    if l > 0
                        && ox >= finer_cx - finer_half
                        && ox + size <= finer_cx + finer_half
                        && oz >= finer_cz - finer_half
                        && oz + size <= finer_cz + finer_half
                    {
                        continue;
                    }

                    self.instances
                        .push(tile_instance(ox, oz, size, l, morph_start, morph_end));
                }
            }
        }

        true
    }
}
