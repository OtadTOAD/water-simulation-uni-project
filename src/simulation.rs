use std::sync::Arc;

use rand_distr::Distribution;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, Queue},
    format::Format,
    image::{ImageDimensions, ImageUsage, StorageImage, view::ImageView},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::Sampler,
    sync::GpuFuture,
};

mod h0_spec_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/h0_spec.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub const TEXTURE_SIZE: u32 = 512;
const WORKGROUP_SIZE: u32 = 16;

fn generate_gaussian_noise(size: u32) -> Vec<[f32; 4]> {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    let mut data = Vec::with_capacity((size * size) as usize);
    for _ in 0..(size * size) {
        let real = normal.sample(&mut rng);
        let imag = normal.sample(&mut rng);
        data.push([real, imag, 0.0, 0.0]);
    }

    data
}

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
    pub noise_image: Arc<ImageView<StorageImage>>,
    pub spec_h0: Arc<ImageView<StorageImage>>,

    h0_spectrum_pipeline: Arc<ComputePipeline>,
}

impl Simulation {
    pub fn new(
        allocator: &StandardMemoryAllocator,
        queue: &Arc<Queue>,
        command_buffer_allocator: &StandardCommandBufferAllocator,
        device: &Arc<Device>,
    ) -> Self {
        let noise_image = Self::generate_noise_texture(allocator, queue, command_buffer_allocator);
        let spec_h0 = create_image(allocator, queue.queue_family_index());

        let h0_shader =
            h0_spec_shader::load(device.clone()).expect("Failed to load test compute shader");
        let h0_spectrum_pipeline = ComputePipeline::new(
            device.clone(),
            h0_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute pipeline");

        Simulation {
            noise_image: ImageView::new_default(noise_image).unwrap(),
            spec_h0,

            h0_spectrum_pipeline,
        }
    }

    pub fn run_h0_spec(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        sampler: Arc<Sampler>,
    ) {
        let pipeline_layout = self.h0_spectrum_pipeline.layout();
        let descriptor_set_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    self.noise_image.clone(),
                    sampler.clone(),
                ),
                WriteDescriptorSet::image_view(1, self.spec_h0.clone()),
            ],
        )
        .expect("Failed to create descriptor set");

        let push_constants = h0_spec_shader::ty::SimParams {
            windDirection: [1.0, 0.0],
            windSpeed: 10.0,
            amplitude: 0.05,
            gridSize: 1000.0,
            gravity: 9.81,
        };

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
            .push_constants(pipeline_layout.clone(), 0, push_constants)
            .dispatch(workgroup_count)
            .expect("Failed to dispatch compute shader");
    }

    fn generate_noise_texture(
        memory_allocator: &StandardMemoryAllocator,
        queue: &Arc<Queue>,
        command_buffer_allocator: &StandardCommandBufferAllocator,
    ) -> Arc<StorageImage> {
        let noise_data = generate_gaussian_noise(TEXTURE_SIZE);

        let noise_image = StorageImage::with_usage(
            memory_allocator,
            ImageDimensions::Dim2d {
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
                array_layers: 1,
            },
            Format::R32G32B32A32_SFLOAT,
            ImageUsage {
                transfer_dst: true,
                storage: true,
                sampled: true,
                ..ImageUsage::empty()
            },
            vulkano::image::ImageCreateFlags::empty(),
            [queue.queue_family_index()],
        )
        .unwrap();

        let staging_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator,
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            },
            false,
            noise_data,
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging_buffer,
                noise_image.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        command_buffer
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        noise_image
    }
}
