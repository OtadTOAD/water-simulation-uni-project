use std::sync::Arc;

use rand_distr::Distribution;
use vulkano::{
    buffer::{BufferContents, BufferUsage, CpuAccessibleBuffer},
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
    shader::ShaderModule,
    sync::GpuFuture,
};

mod init_spec_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/init_spec.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod conj_spec_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/conj_spec.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub const TEXTURE_SIZE: u32 = 512;
const WORKGROUP_SIZE: [u32; 3] = [TEXTURE_SIZE / 8, TEXTURE_SIZE / 8, 1];

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

fn create_pipeline(device: Arc<Device>, shader: Arc<ShaderModule>) -> Arc<ComputePipeline> {
    ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("Failed to create compute pipeline")
}

fn calculate_spectrum_params(wind_speed: f32, fetch: f32, g: f32) -> (f32, f32) {
    let peak_omega = 22.0 * (g * g / (wind_speed * fetch)).powf(1.0 / 3.0);
    let alpha = 0.076 * (g * fetch / (wind_speed * wind_speed)).powf(-0.22);
    (alpha, peak_omega)
}

pub struct Simulation {
    pub noise_image: Arc<ImageView<StorageImage>>,
    pub spec_hk: Arc<ImageView<StorageImage>>,
    pub spec_h0: Arc<ImageView<StorageImage>>,
    pub waves_data: Arc<ImageView<StorageImage>>,

    init_spec_pipeline: Arc<ComputePipeline>,
    conj_spec_pipeline: Arc<ComputePipeline>,

    pub time: f32,
}

impl Simulation {
    pub fn new(
        allocator: &StandardMemoryAllocator,
        queue: &Arc<Queue>,
        command_buffer_allocator: &StandardCommandBufferAllocator,
        device: &Arc<Device>,
    ) -> Self {
        let noise_image = Self::generate_noise_texture(allocator, queue, command_buffer_allocator);
        let waves_data = create_image(allocator, queue.queue_family_index());
        let spec_hk = create_image(allocator, queue.queue_family_index());
        let spec_h0 = create_image(allocator, queue.queue_family_index());

        let init_spec_pipeline = create_pipeline(
            device.clone(),
            init_spec_shader::load(device.clone()).expect("Failed to load init compute shader"),
        );
        let conj_spec_pipeline = create_pipeline(
            device.clone(),
            conj_spec_shader::load(device.clone()).expect("Failed to load conj compute shader"),
        );

        Simulation {
            noise_image: ImageView::new_default(noise_image).unwrap(),
            waves_data,
            spec_hk,
            spec_h0,

            init_spec_pipeline,
            conj_spec_pipeline,

            time: 0.0,
        }
    }

    pub fn run_compute_shader(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        pipeline: Arc<ComputePipeline>,
        bindings: Vec<WriteDescriptorSet>,
        push_constants: impl BufferContents,
    ) {
        let pipeline_layout = pipeline.layout();
        let descriptor_set_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout.clone(),
            bindings,
        )
        .expect("Failed to create descriptor set");

        command_buffer
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_layout.clone(),
                0,
                descriptor_set,
            )
            .push_constants(pipeline_layout.clone(), 0, push_constants)
            .dispatch(WORKGROUP_SIZE)
            .expect("Failed to dispatch compute shader");
    }

    pub fn init(
        &self,
        cmd_alloc: &StandardCommandBufferAllocator,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        queue: Arc<Queue>,
        sampler: Arc<Sampler>,
    ) {
        let mut cmd0 = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let (alpha, peak_omega) = calculate_spectrum_params(0.5, 100000.0, 9.81);
        self.run_compute_shader(
            &mut cmd0,
            descriptor_set_allocator,
            self.init_spec_pipeline.clone(),
            vec![
                WriteDescriptorSet::image_view(0, self.waves_data.clone()),
                WriteDescriptorSet::image_view(1, self.spec_hk.clone()),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    self.noise_image.clone(),
                    sampler.clone(),
                ),
            ],
            init_spec_shader::ty::PushConstants {
                size: TEXTURE_SIZE,
                lengthScale: 250.0,
                cutoffHigh: 9999.0,
                cutoffLow: 0.0001,
                gravityAcceleration: 9.81,
                depth: 500.0,

                scale1: 1.0,
                angle1: (-29.81_f32).to_radians(),
                spreadBlend1: 1.0,
                swell1: 0.198,
                alpha1: alpha,
                peakOmega1: peak_omega,
                gamma1: 3.3,
                shortWavesFade1: 0.01,

                // This is disabled
                scale2: 0.0,
                angle2: 0.0,
                spreadBlend2: 1.0,
                swell2: 1.0,
                alpha2: 0.0081,
                peakOmega2: 0.831,
                gamma2: 3.3,
                shortWavesFade2: 0.01,
            },
        );
        cmd0.build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let mut cmd1 = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        self.run_compute_shader(
            &mut cmd1,
            descriptor_set_allocator,
            self.conj_spec_pipeline.clone(),
            vec![
                WriteDescriptorSet::image_view(0, self.spec_hk.clone()),
                WriteDescriptorSet::image_view(1, self.spec_h0.clone()),
            ],
            conj_spec_shader::ty::PushConstants { size: TEXTURE_SIZE },
        );
        cmd1.build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn run(&self, cmd_alloc: &StandardCommandBufferAllocator, queue: Arc<Queue>) {
        /*
        let mut commands = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        commands
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();*/
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
