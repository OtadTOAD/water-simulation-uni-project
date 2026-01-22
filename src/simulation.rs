use std::sync::Arc;

use rand_distr::Distribution;
use vulkano::{
    buffer::{BufferContents, BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, CopyImageInfo,
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
mod time_spec_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/time_spec.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod fft_init_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/fft_init.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod fft_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/fast_fourier_transform.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod texture_merger_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/texture_merger.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub const TEXTURE_SIZE: u32 = 1024;
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
    let alpha = 0.076 * (g * fetch / (wind_speed * wind_speed)).powf(-0.22);
    let peak_omega = 22.0 * ((wind_speed * fetch) / (g * g)).powf(-0.33);
    (alpha, peak_omega)
}

pub struct Simulation {
    pub noise_image: Arc<ImageView<StorageImage>>,
    pub spec_hk: Arc<ImageView<StorageImage>>,
    pub spec_h0: Arc<ImageView<StorageImage>>,
    pub waves_data: Arc<ImageView<StorageImage>>,

    pub displacement_map: Arc<ImageView<StorageImage>>,
    pub derivatives_map: Arc<ImageView<StorageImage>>,
    pub turbulence_map: Arc<ImageView<StorageImage>>,
    pub camera_depth_map: Arc<ImageView<StorageImage>>,
    pub foam_map: Arc<ImageView<StorageImage>>,

    precomputed_data: Arc<ImageView<StorageImage>>,
    buffer: Arc<ImageView<StorageImage>>,
    dx_dz: Arc<ImageView<StorageImage>>,
    dy_dxz: Arc<ImageView<StorageImage>>,
    dyx_dyz: Arc<ImageView<StorageImage>>,
    dxx_dzz: Arc<ImageView<StorageImage>>,

    fft_init_pipeline: Arc<ComputePipeline>,
    fft_pipeline: Arc<ComputePipeline>,

    init_spec_pipeline: Arc<ComputePipeline>,
    conj_spec_pipeline: Arc<ComputePipeline>,
    time_spec_pipeline: Arc<ComputePipeline>,
    texture_merger_pipeline: Arc<ComputePipeline>,
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

        let displacement_map = create_image(allocator, queue.queue_family_index());
        let derivatives_map = create_image(allocator, queue.queue_family_index());
        let turbulence_map = create_image(allocator, queue.queue_family_index());
        let camera_depth_map = create_image(allocator, queue.queue_family_index());
        let foam_map = create_image(allocator, queue.queue_family_index());

        let precomputed_data = create_image(allocator, queue.queue_family_index());
        let buffer = create_image(allocator, queue.queue_family_index());
        let dx_dz = create_image(allocator, queue.queue_family_index());
        let dy_dxz = create_image(allocator, queue.queue_family_index());
        let dyx_dyz = create_image(allocator, queue.queue_family_index());
        let dxx_dzz = create_image(allocator, queue.queue_family_index());

        let init_spec_pipeline = create_pipeline(
            device.clone(),
            init_spec_shader::load(device.clone()).expect("Failed to load init compute shader"),
        );
        let conj_spec_pipeline = create_pipeline(
            device.clone(),
            conj_spec_shader::load(device.clone()).expect("Failed to load conj compute shader"),
        );
        let time_spec_pipeline = create_pipeline(
            device.clone(),
            time_spec_shader::load(device.clone()).expect("Failed to load time compute shader"),
        );

        let fft_init_pipeline = create_pipeline(
            device.clone(),
            fft_init_shader::load(device.clone()).expect("Failed to load fft compute shader"),
        );
        let fft_pipeline = create_pipeline(
            device.clone(),
            fft_shader::load(device.clone()).expect("Failed to load fft compute shader"),
        );

        let texture_merger_pipeline = create_pipeline(
            device.clone(),
            texture_merger_shader::load(device.clone())
                .expect("Failed to load texture merger compute shader"),
        );

        Simulation {
            noise_image: ImageView::new_default(noise_image).unwrap(),
            waves_data,
            spec_hk,
            spec_h0,

            displacement_map,
            derivatives_map,
            turbulence_map,
            camera_depth_map,
            foam_map,

            precomputed_data,
            buffer,
            dx_dz,
            dy_dxz,
            dyx_dyz,
            dxx_dzz,

            fft_init_pipeline,
            fft_pipeline,

            init_spec_pipeline,
            conj_spec_pipeline,
            time_spec_pipeline,
            texture_merger_pipeline,

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

        let wind_speed = 0.5;
        let (alpha, peak_omega) = calculate_spectrum_params(wind_speed, 100000.0, 9.81);

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
                lengthScale: 100.0,
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
        self.run_compute_shader(
            &mut cmd0,
            descriptor_set_allocator,
            self.fft_init_pipeline.clone(),
            vec![WriteDescriptorSet::image_view(
                0,
                self.precomputed_data.clone(),
            )],
            fft_init_shader::ty::PushConstants { size: TEXTURE_SIZE },
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

    pub fn run(
        &self,
        cmd_alloc: &StandardCommandBufferAllocator,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        queue: Arc<Queue>,
    ) {
        let mut cmd0 = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        self.run_compute_shader(
            &mut cmd0,
            descriptor_set_allocator,
            self.time_spec_pipeline.clone(),
            vec![
                WriteDescriptorSet::image_view(0, self.waves_data.clone()),
                WriteDescriptorSet::image_view(1, self.spec_h0.clone()),
                // Displacement
                WriteDescriptorSet::image_view(2, self.dx_dz.clone()),
                WriteDescriptorSet::image_view(3, self.dy_dxz.clone()),
                WriteDescriptorSet::image_view(4, self.dyx_dyz.clone()),
                WriteDescriptorSet::image_view(5, self.dxx_dzz.clone()),
            ],
            time_spec_shader::ty::PushConstants {
                size: TEXTURE_SIZE,
                time: self.time,
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

        self.run_ifft_2d(
            cmd_alloc,
            descriptor_set_allocator,
            queue.clone(),
            true,
            false,
            true,
            self.dx_dz.clone(),
            self.buffer.clone(),
        );
        self.run_ifft_2d(
            cmd_alloc,
            descriptor_set_allocator,
            queue.clone(),
            true,
            false,
            true,
            self.dy_dxz.clone(),
            self.buffer.clone(),
        );
        self.run_ifft_2d(
            cmd_alloc,
            descriptor_set_allocator,
            queue.clone(),
            true,
            false,
            true,
            self.dyx_dyz.clone(),
            self.buffer.clone(),
        );
        self.run_ifft_2d(
            cmd_alloc,
            descriptor_set_allocator,
            queue.clone(),
            true,
            false,
            true,
            self.dxx_dzz.clone(),
            self.buffer.clone(),
        );

        let mut cmd1 = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        self.run_compute_shader(
            &mut cmd1,
            descriptor_set_allocator,
            self.texture_merger_pipeline.clone(),
            vec![
                WriteDescriptorSet::image_view(0, self.displacement_map.clone()),
                WriteDescriptorSet::image_view(1, self.derivatives_map.clone()),
                WriteDescriptorSet::image_view(2, self.turbulence_map.clone()),
                // Displacement
                WriteDescriptorSet::image_view(3, self.dx_dz.clone()),
                WriteDescriptorSet::image_view(4, self.dy_dxz.clone()),
                WriteDescriptorSet::image_view(5, self.dyx_dyz.clone()),
                WriteDescriptorSet::image_view(6, self.dxx_dzz.clone()),
            ],
            texture_merger_shader::ty::PushConstants {
                size: TEXTURE_SIZE,
                dlt: self.time,
            },
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

    fn run_ifft_2d(
        &self,
        cmd_alloc: &StandardCommandBufferAllocator,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        queue: Arc<Queue>,
        output_to_input: bool,
        scale: bool,
        permute: bool,
        input: Arc<ImageView<StorageImage>>,
        buffer: Arc<ImageView<StorageImage>>,
    ) {
        let log_size = (TEXTURE_SIZE as f32).log2() as u32;
        let mut ping_pong = 0;

        let mut commands = AutoCommandBufferBuilder::primary(
            cmd_alloc,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        for i in 0..log_size {
            ping_pong ^= 1;

            self.run_compute_shader(
                &mut commands,
                descriptor_set_allocator,
                self.fft_pipeline.clone(),
                vec![
                    WriteDescriptorSet::image_view(0, self.precomputed_data.clone()),
                    WriteDescriptorSet::image_view(1, input.clone()),
                    WriteDescriptorSet::image_view(2, buffer.clone()),
                ],
                fft_shader::ty::PushConstants {
                    size: TEXTURE_SIZE,
                    stage: i,
                    ping_pong,
                    mode: 2, // Inverse Horizontal pass
                },
            );

            commands.dispatch(WORKGROUP_SIZE).unwrap();
        }

        for i in 0..log_size {
            ping_pong ^= 1;

            self.run_compute_shader(
                &mut commands,
                descriptor_set_allocator,
                self.fft_pipeline.clone(),
                vec![
                    WriteDescriptorSet::image_view(0, self.precomputed_data.clone()),
                    WriteDescriptorSet::image_view(1, input.clone()),
                    WriteDescriptorSet::image_view(2, buffer.clone()),
                ],
                fft_shader::ty::PushConstants {
                    size: TEXTURE_SIZE,
                    stage: i,
                    ping_pong,
                    mode: 3, // Inverse Vertical pass
                },
            );

            commands.dispatch(WORKGROUP_SIZE).unwrap();
        }

        if ping_pong == 1 && output_to_input {
            commands
                .copy_image(CopyImageInfo::images(
                    buffer.image().clone(),
                    input.image().clone(),
                ))
                .unwrap();
        }
        if ping_pong == 0 && !output_to_input {
            commands
                .copy_image(CopyImageInfo::images(
                    input.image().clone(),
                    buffer.image().clone(),
                ))
                .unwrap();
        }

        if permute {
            self.run_compute_shader(
                &mut commands,
                descriptor_set_allocator,
                self.fft_pipeline.clone(),
                vec![
                    WriteDescriptorSet::image_view(0, self.precomputed_data.clone()),
                    WriteDescriptorSet::image_view(1, input.clone()),
                    WriteDescriptorSet::image_view(2, buffer.clone()),
                ],
                fft_shader::ty::PushConstants {
                    size: TEXTURE_SIZE,
                    stage: 0,
                    ping_pong,
                    mode: 5, // Permute pass
                },
            );
        }
        if scale {
            self.run_compute_shader(
                &mut commands,
                descriptor_set_allocator,
                self.fft_pipeline.clone(),
                vec![
                    WriteDescriptorSet::image_view(0, self.precomputed_data.clone()),
                    WriteDescriptorSet::image_view(1, input.clone()),
                    WriteDescriptorSet::image_view(2, buffer.clone()),
                ],
                fft_shader::ty::PushConstants {
                    size: TEXTURE_SIZE,
                    stage: 0,
                    ping_pong,
                    mode: 4, // Scale pass
                },
            );
        }

        commands
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
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
