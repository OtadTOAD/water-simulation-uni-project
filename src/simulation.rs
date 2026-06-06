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
mod fft_inv_horizontal_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/fft_inv_horizontal.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod fft_inv_vertical_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/fft_inv_vertical.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod fft_permute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/fft_permute.comp",
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
    let alpha = 0.076 * (g * fetch / (wind_speed * wind_speed)).powf(-0.22);
    let peak_omega = 22.0 * ((wind_speed * fetch) / (g * g)).powf(-0.33);
    (alpha, peak_omega)
}

struct IfftSets {
    h_even: Arc<PersistentDescriptorSet>,
    h_odd: Arc<PersistentDescriptorSet>,
    v_even: Arc<PersistentDescriptorSet>,
    v_odd: Arc<PersistentDescriptorSet>,
    permute: Arc<PersistentDescriptorSet>,
}

struct RunSets {
    time_spec: Arc<PersistentDescriptorSet>,
    ifft: [IfftSets; 4],
    merger: Arc<PersistentDescriptorSet>,
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
    fft_inv_horizontal_pipeline: Arc<ComputePipeline>,
    fft_inv_vertical_pipeline: Arc<ComputePipeline>,
    fft_permute_pipeline: Arc<ComputePipeline>,

    init_spec_pipeline: Arc<ComputePipeline>,
    conj_spec_pipeline: Arc<ComputePipeline>,
    time_spec_pipeline: Arc<ComputePipeline>,
    texture_merger_pipeline: Arc<ComputePipeline>,
    sets: Option<RunSets>,
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
        let fft_inv_horizontal_pipeline = create_pipeline(
            device.clone(),
            fft_inv_horizontal_shader::load(device.clone())
                .expect("Failed to load fft inv horizontal compute shader"),
        );
        let fft_inv_vertical_pipeline = create_pipeline(
            device.clone(),
            fft_inv_vertical_shader::load(device.clone())
                .expect("Failed to load fft inv vertical compute shader"),
        );
        let fft_permute_pipeline = create_pipeline(
            device.clone(),
            fft_permute_shader::load(device.clone())
                .expect("Failed to load fft permute compute shader"),
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
            fft_inv_horizontal_pipeline,
            fft_inv_vertical_pipeline,
            fft_permute_pipeline,

            init_spec_pipeline,
            conj_spec_pipeline,
            time_spec_pipeline,
            texture_merger_pipeline,

            sets: None,
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
        &mut self,
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

        let wind_speed = 0.25;
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
                spreadBlend1: 0.95,
                swell1: 0.198,
                alpha1: alpha,
                peakOmega1: peak_omega,
                gamma1: 3.3,
                shortWavesFade1: 0.01,

                scale2: 0.5,
                angle2: (-5.81_f32).to_radians(),
                spreadBlend2: 0.9,
                swell2: 0.2,
                alpha2: alpha,
                peakOmega2: peak_omega,
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

        self.sets = Some(self.build_run_sets(descriptor_set_allocator));
    }

    fn build_run_sets(&self, alloc: &StandardDescriptorSetAllocator) -> RunSets {
        RunSets {
            time_spec: Self::make_set(
                alloc,
                &self.time_spec_pipeline,
                vec![
                    WriteDescriptorSet::image_view(0, self.waves_data.clone()),
                    WriteDescriptorSet::image_view(1, self.spec_h0.clone()),
                    WriteDescriptorSet::image_view(2, self.dx_dz.clone()),
                    WriteDescriptorSet::image_view(3, self.dy_dxz.clone()),
                    WriteDescriptorSet::image_view(4, self.dyx_dyz.clone()),
                    WriteDescriptorSet::image_view(5, self.dxx_dzz.clone()),
                ],
            ),
            ifft: [
                self.build_ifft_sets(alloc, &self.dx_dz),
                self.build_ifft_sets(alloc, &self.dy_dxz),
                self.build_ifft_sets(alloc, &self.dyx_dyz),
                self.build_ifft_sets(alloc, &self.dxx_dzz),
            ],
            merger: Self::make_set(
                alloc,
                &self.texture_merger_pipeline,
                vec![
                    WriteDescriptorSet::image_view(0, self.displacement_map.clone()),
                    WriteDescriptorSet::image_view(1, self.derivatives_map.clone()),
                    WriteDescriptorSet::image_view(2, self.turbulence_map.clone()),
                    WriteDescriptorSet::image_view(3, self.dx_dz.clone()),
                    WriteDescriptorSet::image_view(4, self.dy_dxz.clone()),
                    WriteDescriptorSet::image_view(5, self.dyx_dyz.clone()),
                    WriteDescriptorSet::image_view(6, self.dxx_dzz.clone()),
                ],
            ),
        }
    }

    fn build_ifft_sets(
        &self,
        alloc: &StandardDescriptorSetAllocator,
        target: &Arc<ImageView<StorageImage>>,
    ) -> IfftSets {
        let scratch = &self.buffer;
        let read_target = |read: &Arc<ImageView<StorageImage>>,
                           write: &Arc<ImageView<StorageImage>>| {
            vec![
                WriteDescriptorSet::image_view(0, self.precomputed_data.clone()),
                WriteDescriptorSet::image_view(1, read.clone()),
                WriteDescriptorSet::image_view(2, write.clone()),
            ]
        };
        IfftSets {
            h_even: Self::make_set(alloc, &self.fft_inv_horizontal_pipeline, read_target(target, scratch)),
            h_odd: Self::make_set(alloc, &self.fft_inv_horizontal_pipeline, read_target(scratch, target)),
            v_even: Self::make_set(alloc, &self.fft_inv_vertical_pipeline, read_target(target, scratch)),
            v_odd: Self::make_set(alloc, &self.fft_inv_vertical_pipeline, read_target(scratch, target)),
            permute: Self::make_set(
                alloc,
                &self.fft_permute_pipeline,
                vec![WriteDescriptorSet::image_view(0, target.clone())],
            ),
        }
    }

    fn make_set(
        alloc: &StandardDescriptorSetAllocator,
        pipeline: &Arc<ComputePipeline>,
        writes: Vec<WriteDescriptorSet>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        PersistentDescriptorSet::new(alloc, layout.clone(), writes)
            .expect("Failed to create descriptor set")
    }

    pub fn record(&self, cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        let sets = self
            .sets
            .as_ref()
            .expect("Simulation::init must run before record");
        let log_size = (TEXTURE_SIZE as f32).log2() as u32;

        Self::dispatch(
            cmd,
            &self.time_spec_pipeline,
            &sets.time_spec,
            time_spec_shader::ty::PushConstants {
                size: TEXTURE_SIZE,
                time: self.time,
            },
        );

        for target in &sets.ifft {
            for stage in 0..log_size {
                let set = if stage % 2 == 0 { &target.h_even } else { &target.h_odd };
                Self::dispatch(
                    cmd,
                    &self.fft_inv_horizontal_pipeline,
                    set,
                    fft_inv_horizontal_shader::ty::PushConstants {
                        size: TEXTURE_SIZE,
                        stage,
                    },
                );
            }
            for stage in 0..log_size {
                // If you don't continue it, data ends up in wrong buffer
                let set = if (stage + log_size) % 2 == 0 { &target.v_even } else { &target.v_odd };
                Self::dispatch(
                    cmd,
                    &self.fft_inv_vertical_pipeline,
                    set,
                    fft_inv_vertical_shader::ty::PushConstants {
                        size: TEXTURE_SIZE,
                        stage,
                    },
                );
            }
            Self::dispatch(
                cmd,
                &self.fft_permute_pipeline,
                &target.permute,
                fft_permute_shader::ty::PushConstants { size: TEXTURE_SIZE },
            );
        }

        Self::dispatch(
            cmd,
            &self.texture_merger_pipeline,
            &sets.merger,
            texture_merger_shader::ty::PushConstants {
                size: TEXTURE_SIZE,
                dlt: self.time,
            },
        );
    }

    fn dispatch(
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<ComputePipeline>,
        set: &Arc<PersistentDescriptorSet>,
        push_constants: impl BufferContents,
    ) {
        cmd.bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch(WORKGROUP_SIZE)
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
