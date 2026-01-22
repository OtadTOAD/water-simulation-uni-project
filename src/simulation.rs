use std::sync::Arc;

use rand_distr::Distribution;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract, allocator::StandardCommandBufferAllocator,
    },
    device::Queue,
    format::Format,
    image::{ImageDimensions, ImageUsage, StorageImage, view::ImageView},
    memory::allocator::StandardMemoryAllocator,
    sync::GpuFuture,
};

const TEXTURE_SIZE: u32 = 256;

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

pub struct Simulation {
    pub noise_image: Arc<ImageView<StorageImage>>,
}

impl Simulation {
    pub fn new(
        memory_allocator: &StandardMemoryAllocator,
        queue: &Arc<Queue>,
        command_buffer_allocator: &StandardCommandBufferAllocator,
    ) -> Self {
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

        let noise_image_view = ImageView::new_default(noise_image).unwrap();
        Simulation {
            noise_image: noise_image_view,
        }
    }
}
