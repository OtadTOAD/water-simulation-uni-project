use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::Device,
    format::Format,
    image::{ImageDimensions, StorageImage, view::ImageView},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

mod h0_spectrum_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/render/shaders/h0_spectrum.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod ht_spectrum_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/render/shaders/ht_spectrum.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub const TEXTURE_SIZE: u32 = 256; // Must be power of 2 for FFT
const WORKGROUP_SIZE: u32 = 16; // Must divide TEXTURE_SIZE evenly(MUST BE same as one in compute shader)

fn create_image(
    allocator: &StandardMemoryAllocator,
    family_idx: u32,
) -> Arc<ImageView<StorageImage>> {
    let img = StorageImage::new(
        allocator,
        ImageDimensions::Dim2d {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            array_layers: 1,
        },
        Format::R32G32B32A32_SFLOAT,
        [family_idx],
    )
    .unwrap();
    ImageView::new_default(img).unwrap()
}

pub struct Simulation {
    pub spec_h0: Arc<ImageView<StorageImage>>,
    pub spec_ht: Arc<ImageView<StorageImage>>,
    pub spec_temp: Arc<ImageView<StorageImage>>,
    pub map_displace: Arc<ImageView<StorageImage>>,
    pub map_normal: Arc<ImageView<StorageImage>>,

    time_buffer: Arc<CpuAccessibleBuffer<ht_spectrum_shader::ty::PushConstants>>,
    param_buffer: Arc<CpuAccessibleBuffer<h0_spectrum_shader::ty::SimParams>>,

    ht_spectrum_pipeline: Arc<ComputePipeline>,
    h0_spectrum_pipeline: Arc<ComputePipeline>,
}

impl Simulation {
    pub fn new(
        allocator: &StandardMemoryAllocator,
        device: Arc<Device>, // <-- Added device parameter
        family_idx: u32,
    ) -> Self {
        let spec_h0 = create_image(allocator, family_idx);
        let spec_ht = create_image(allocator, family_idx);
        let spec_temp = create_image(allocator, family_idx);
        let map_displace = create_image(allocator, family_idx);
        let map_normal = create_image(allocator, family_idx);

        let h0_shader =
            h0_spectrum_shader::load(device.clone()).expect("Failed to load test compute shader");
        let h0_spectrum_pipeline = ComputePipeline::new(
            device.clone(),
            h0_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute pipeline");

        let ht_shader =
            ht_spectrum_shader::load(device.clone()).expect("Failed to load ht compute shader");
        let ht_spectrum_pipeline = ComputePipeline::new(
            device.clone(),
            ht_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute pipeline");

        let param_buffer = CpuAccessibleBuffer::from_data(
            allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            h0_spectrum_shader::ty::SimParams {
                windDirection: [1.0, 0.0],
                windSpeed: 15.0,
                amplitude: 1.0,
                gridSize: 1000.0,
                gravity: 9.81,
            },
        )
        .unwrap();

        let time_buffer = CpuAccessibleBuffer::from_data(
            allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            ht_spectrum_shader::ty::PushConstants { time: 0.0 },
        )
        .unwrap();

        Simulation {
            spec_h0,
            spec_ht,
            spec_temp,
            map_displace,
            map_normal,

            h0_spectrum_pipeline,
            ht_spectrum_pipeline,

            time_buffer,
            param_buffer,
        }
    }

    // TODO: This should only run when params change(But for now I am too lazy so it runs every frame)
    pub fn generate_h0_spectrum(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
    ) {
        // TODO: Better to keep descriptor sets around instead of recreating each frame
        let pipeline_layout = self.h0_spectrum_pipeline.layout();
        let descriptor_set_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.spec_h0.clone()),
                WriteDescriptorSet::buffer(1, self.param_buffer.clone()),
            ],
        )
        .expect("Failed to create descriptor set");

        let workgroup_count = [
            TEXTURE_SIZE / WORKGROUP_SIZE,
            TEXTURE_SIZE / WORKGROUP_SIZE,
            1,
        ];
        command_buffer
            .bind_pipeline_compute(self.h0_spectrum_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_layout.clone(),
                0,
                descriptor_set,
            )
            .dispatch(workgroup_count)
            .expect("Failed to dispatch compute shader");
    }

    pub fn update_time_spectrum(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        delta: f32,
    ) {
        // Is this best way of doing this? Probably not, but I don't care for now
        match self.time_buffer.write() {
            Ok(mut content) => {
                content.time += delta;
            }
            Err(_) => {}
        }

        // TODO: Better to keep descriptor sets around instead of recreating each frame
        let pipeline_layout = self.ht_spectrum_pipeline.layout();
        let descriptor_set_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.spec_h0.clone()),
                WriteDescriptorSet::image_view(1, self.spec_ht.clone()),
                WriteDescriptorSet::buffer(2, self.time_buffer.clone()),
            ],
        )
        .expect("Failed to create descriptor set");

        let workgroup_count = [
            TEXTURE_SIZE / WORKGROUP_SIZE,
            TEXTURE_SIZE / WORKGROUP_SIZE,
            1,
        ];
        command_buffer
            .bind_pipeline_compute(self.ht_spectrum_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_layout.clone(),
                0,
                descriptor_set,
            )
            .dispatch(workgroup_count)
            .expect("Failed to dispatch compute shader");
    }
}
