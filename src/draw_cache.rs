use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    memory::allocator::StandardMemoryAllocator,
    pipeline::{GraphicsPipeline, Pipeline},
};

use crate::instance::{Instance, Mesh, Vertex};

pub struct DrawCache {
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub inst_buffer: Arc<CpuAccessibleBuffer<[Instance]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    pub geometry_sets: Vec<Arc<PersistentDescriptorSet>>,
}

impl DrawCache {
    pub fn new(
        mesh: &Mesh,
        instances: &Vec<Instance>,
        memory_allocator: &StandardMemoryAllocator,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        geometry_pipeline: &Arc<GraphicsPipeline>,
        descriptor_writes: Vec<impl IntoIterator<Item = WriteDescriptorSet>>,
    ) -> Self {
        let inst_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            instances.iter().cloned(),
        )
        .unwrap();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            mesh.vertices.iter().cloned(),
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator,
            BufferUsage {
                index_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            mesh.indices.iter().cloned(),
        )
        .unwrap();

        let mut geometry_sets = Vec::new();
        for writes in descriptor_writes {
            let geometry_layout = geometry_pipeline
                .layout()
                .set_layouts()
                .get(geometry_sets.len() as usize)
                .unwrap();
            let geometry_set = PersistentDescriptorSet::new(
                descriptor_set_allocator,
                geometry_layout.clone(),
                writes,
            )
            .unwrap();
            geometry_sets.push(geometry_set);
        }

        DrawCache {
            geometry_sets,
            index_buffer,
            vertex_buffer,
            inst_buffer,
        }
    }
}
