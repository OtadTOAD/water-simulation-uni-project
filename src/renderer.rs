use std::{mem, sync::Arc};

use vulkano::{
    VulkanLibrary,
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        self, Device, DeviceCreateInfo, Features, Queue, QueueCreateInfo,
        physical::PhysicalDeviceType,
    },
    format::Format,
    image::{
        AttachmentImage, ImageAccess, ImageDimensions, ImmutableImage, MipmapsCount,
        SwapchainImage, view::ImageView,
    },
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint,
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{
        Filter, LOD_CLAMP_NONE, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
    },
    swapchain::{
        self, AcquireError, PresentMode, Surface, Swapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::window::{Window, WindowBuilder};

use crate::{
    camera::Camera,
    draw_cache::DrawCache,
    instance::{Instance, Mesh, Vertex},
    simulation::{CASCADE_LENGTH_SCALES, Simulation},
};

vulkano::impl_vertex!(Vertex, position, uv);
vulkano::impl_vertex!(Instance, instance_model, instance_normal, lod_morph);

mod water_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/water.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod water_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/water.frag",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod sky_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/sky.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
mod sky_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/sky.frag",
    }
}

fn get_window(surface: &Arc<Surface>) -> &Window {
    surface.object().unwrap().downcast_ref::<Window>().unwrap()
}

fn required_surface_extensions(library: &VulkanLibrary) -> vulkano::instance::InstanceExtensions {
    let ideal = vulkano::instance::InstanceExtensions {
        khr_surface: true,
        khr_xlib_surface: true,
        khr_xcb_surface: true,
        khr_wayland_surface: true,
        khr_win32_surface: true,
        khr_get_surface_capabilities2: true,
        khr_get_physical_device_properties2: true,
        ..vulkano::instance::InstanceExtensions::empty()
    };
    library.supported_extensions().intersection(&ideal)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn create_surface(window: Arc<Window>, instance: Arc<vulkano::instance::Instance>) -> Arc<Surface> {
    use winit::platform::unix::WindowExtUnix;
    unsafe {
        match (window.wayland_display(), window.wayland_surface()) {
            (Some(display), Some(surface)) => {
                Surface::from_wayland(instance, display, surface, Some(window)).unwrap()
            }
            _ => Surface::from_xlib(
                instance,
                window.xlib_display().unwrap(),
                window.xlib_window().unwrap() as _,
                Some(window),
            )
            .unwrap(),
        }
    }
}

#[cfg(target_os = "windows")]
fn create_surface(window: Arc<Window>, instance: Arc<vulkano::instance::Instance>) -> Arc<Surface> {
    use winit::platform::windows::WindowExtWindows;
    unsafe {
        Surface::from_win32(
            instance,
            window.hinstance() as *const (),
            window.hwnd() as *const (),
            Some(window),
        )
        .unwrap()
    }
}

fn load_sky_texture(
    allocator: &StandardMemoryAllocator,
    queue: &Arc<Queue>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
) -> Arc<ImageView<ImmutableImage>> {
    use exr::prelude::*;

    // Read the equirectangular HDR sky into a flat RGBA32F buffer.
    let image = read_first_rgba_layer_from_file(
        "assets/puresky.exr",
        |resolution: Vec2<usize>, _| -> (usize, Vec<[f32; 4]>) {
            (
                resolution.width(),
                vec![[0.0, 0.0, 0.0, 1.0]; resolution.width() * resolution.height()],
            )
        },
        |(width, pixels): &mut (usize, Vec<[f32; 4]>),
         position: Vec2<usize>,
         (r, g, b, a): (f32, f32, f32, f32)| {
            let idx = position.y() * *width + position.x();
            pixels[idx] = [r, g, b, a];
        },
    )
    .expect("Failed to read assets/puresky.exr");

    let (width, pixels) = image.layer_data.channel_data.pixels;
    let height = pixels.len() / width;

    let mut uploader = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let sky_image = ImmutableImage::from_iter(
        allocator,
        pixels,
        ImageDimensions::Dim2d {
            width: width as u32,
            height: height as u32,
            array_layers: 1,
        },
        MipmapsCount::Log2,
        Format::R32G32B32A32_SFLOAT,
        &mut uploader,
    )
    .unwrap();

    uploader
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    ImageView::new_default(sky_image).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderStage {
    Stopped,
    Render,
    NeedsRedraw,
}

pub struct Renderer {
    pub device: Arc<Device>,
    pub aspect_ratio: f32,

    surface: Arc<Surface>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    render_pass: Arc<RenderPass>,
    geometry_pipeline: Arc<GraphicsPipeline>,
    sky_pipeline: Arc<GraphicsPipeline>,
    sky_set: Arc<PersistentDescriptorSet>,
    sky_push: sky_vert::ty::SkyCamera,
    viewport: Viewport,
    framebuffers: Vec<Arc<Framebuffer>>,
    render_stage: RenderStage,
    commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    image_index: u32,
    acquire_future: Option<SwapchainAcquireFuture>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,

    pub ocean_params_buffer: Arc<CpuAccessibleBuffer<water_frag::ty::OceanParams>>,
    pub mat_params_buffer: Arc<CpuAccessibleBuffer<water_frag::ty::MaterialParams>>,

    pub texture_sampler: Arc<Sampler>,
    pub sky_image: Arc<ImageView<ImmutableImage>>,
    pub sky_sampler: Arc<Sampler>,
    camera_push: water_vert::ty::Camera,
    pub simulation: Simulation,
}

impl Renderer {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        let instance = {
            let library = VulkanLibrary::new().unwrap();

            let extensions = required_surface_extensions(&library);

            vulkano::instance::Instance::new(
                library,
                vulkano::instance::InstanceCreateInfo {
                    enabled_extensions: extensions,
                    enumerate_portability: true,
                    max_api_version: Some(vulkano::Version::V1_1),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(winit::dpi::PhysicalSize::new(800u32, 800u32))
                .build(event_loop)
                .unwrap(),
        );
        let surface = create_surface(window, instance.clone());
        let device_extensions = device::DeviceExtensions {
            ext_full_screen_exclusive: false,
            khr_swapchain: true,
            ..device::DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.graphics
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("No suitable physical device found");

        let anisotropy_supported = physical_device.supported_features().sampler_anisotropy;
        let max_anisotropy = physical_device.properties().max_sampler_anisotropy;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: Features {
                    sampler_anisotropy: anisotropy_supported,
                    ..Features::empty()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();
        let (swapchain, images) = {
            let caps = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let usage = caps.supported_usage_flags;
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );

            let window = get_window(&surface);
            let image_extent: [u32; 2] = window.inner_size().into();

            let present_mode = device
                .physical_device()
                .surface_present_modes(&surface)
                .unwrap()
                .find(|&mode| mode == PresentMode::Mailbox)
                .unwrap_or(PresentMode::Fifo);

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format,
                    image_extent,
                    present_mode,
                    image_usage: usage,
                    composite_alpha: alpha,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = vulkano::ordered_passes_renderpass!(device.clone(),
            attachments: {
                final_color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            passes: [
                {
                    color: [final_color],
                    depth_stencil: {depth},
                    input: []
                }
            ]
        )
        .unwrap();

        let deferred_vert = water_vert::load(device.clone()).unwrap();
        let deferred_frag = water_frag::load(device.clone()).unwrap();
        let geometry_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let geometry_pipeline = GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .instance::<Instance>(),
            )
            .vertex_shader(deferred_vert.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(deferred_frag.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .render_pass(geometry_pass.clone())
            .build(device.clone())
            .unwrap();

        let sky_vert_shader = sky_vert::load(device.clone()).unwrap();
        let sky_frag_shader = sky_frag::load(device.clone()).unwrap();
        let sky_pipeline = GraphicsPipeline::start()
            .vertex_shader(sky_vert_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(sky_frag_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::disabled())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .render_pass(geometry_pass.clone())
            .build(device.clone())
            .unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        let framebuffers = Renderer::window_size_dependent_setup(
            &memory_allocator,
            &images,
            render_pass.clone(),
            &mut viewport,
        );

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let acquire_future = None;
        let commands = None;
        let render_stage = RenderStage::Stopped;
        let image_index = 0;

        let aspect_ratio = {
            let window = get_window(&surface);
            window.inner_size().width as f32 / window.inner_size().height as f32
        };

        let camera_push = water_vert::ty::Camera {
            proj: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            pos: [0.0; 3],
        };

        let texture_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let sky_image = load_sky_texture(&memory_allocator, &queue, &command_buffer_allocator);
        let sky_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [
                    SamplerAddressMode::Repeat,      // longitude wraps around
                    SamplerAddressMode::ClampToEdge, // latitude clamps at the poles
                    SamplerAddressMode::ClampToEdge,
                ],
                mipmap_mode: SamplerMipmapMode::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                anisotropy: if anisotropy_supported {
                    Some(max_anisotropy.min(16.0))
                } else {
                    None
                },
                ..Default::default()
            },
        )
        .unwrap();

        let sky_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            sky_pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                sky_image.clone(),
                sky_sampler.clone(),
            )],
        )
        .unwrap();

        let sky_push = sky_vert::ty::SkyCamera {
            invViewProj: [[0.0; 4]; 4],
            pos: [0.0; 3],
        };

        let simulation = Simulation::new(
            &memory_allocator,
            &queue,
            &command_buffer_allocator,
            &device,
        );

        let ocean_params_buffer = CpuAccessibleBuffer::from_data(
            &memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            water_frag::ty::OceanParams {
                lengthScales: [
                    CASCADE_LENGTH_SCALES[0],
                    CASCADE_LENGTH_SCALES[1],
                    CASCADE_LENGTH_SCALES[2],
                    0.0,
                ],
                lodScale: 1.0,
                sssBase: -0.1,
                sssScale: 12.0,
            },
        )
        .unwrap();
        let mat_params_buffer = CpuAccessibleBuffer::from_data(
            &memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            water_frag::ty::MaterialParams {
                color: [0.03457636, 0.12297464, 0.1981132, 1.0],
                foamColor: [1.0, 1.0, 1.0, 1.0],
                sssColor: [0.1, 0.45, 0.42, 1.0],
                sssStrength: 0.05,
                roughness: 0.311,
                roughnessScale: 0.0044,
                maxGloss: 0.91,
                foamBias: 0.84,
                foamScale: 2.4,
                contactFoam: 1.0,
                time: 0.0,
                lightDir: [0.0, 1.0, 0.0],
            },
        )
        .unwrap();

        Renderer {
            surface,
            device,
            queue,
            swapchain,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            render_pass,
            geometry_pipeline,
            sky_pipeline,
            sky_set,
            sky_push,
            viewport,
            framebuffers,
            render_stage,
            commands,
            image_index,
            acquire_future,

            ocean_params_buffer,
            mat_params_buffer,

            texture_sampler,
            sky_image,
            sky_sampler,
            camera_push,
            aspect_ratio,
            simulation,
        }
    }

    pub fn init(&mut self) {
        self.simulation.init(
            &self.command_buffer_allocator,
            &self.descriptor_set_allocator,
            self.queue.clone(),
            self.texture_sampler.clone(),
        );
    }

    pub fn run_sim(&mut self, delta_time: f32) {
        self.simulation.time += delta_time;
    }

    pub fn window(&self) -> &Window {
        get_window(&self.surface)
    }

    // TODO: This can either be done as multiple smaller buffers
    // Or just use push constants
    pub fn set_camera(&mut self, camera: &Camera) {
        self.camera_push = water_vert::ty::Camera {
            proj: camera.projection_matrix_raw(),
            view: camera.view_matrix_raw(),
            pos: camera.position.into(),
        };
        self.sky_push = sky_vert::ty::SkyCamera {
            invViewProj: camera.inv_view_proj_raw(),
            pos: camera.position.into(),
        };
    }

    pub fn get_draw_cache(
        &self,
        mesh: &Mesh,
        instances: &Vec<Instance>,
        descriptor_writes: Vec<impl IntoIterator<Item = WriteDescriptorSet>>,
    ) -> DrawCache {
        DrawCache::new(
            mesh,
            instances,
            &self.memory_allocator,
            &self.descriptor_set_allocator,
            &self.geometry_pipeline,
            descriptor_writes,
        )
    }

    pub fn update_instances(&self, cache: &mut DrawCache, instances: &[Instance]) {
        cache.update_instances(&self.memory_allocator, instances);
    }

    pub fn recreate_swapchain(&mut self) {
        self.render_stage = RenderStage::NeedsRedraw;
        self.commands = None;

        let window = self
            .surface
            .object()
            .unwrap()
            .downcast_ref::<Window>()
            .unwrap();
        let image_extent: [u32; 2] = window.inner_size().into();
        if image_extent[0] == 0 || image_extent[1] == 0 {
            return;
        }

        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent,
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };

        let new_framebuffers = Renderer::window_size_dependent_setup(
            &self.memory_allocator,
            &new_images,
            self.render_pass.clone(),
            &mut self.viewport,
        );

        let aspect_ratio = window.inner_size().width as f32 / window.inner_size().height as f32;

        self.swapchain = new_swapchain;
        self.framebuffers = new_framebuffers;
        self.render_stage = RenderStage::Stopped;
        self.aspect_ratio = aspect_ratio;
    }

    fn window_size_dependent_setup(
        allocator: &StandardMemoryAllocator,
        images: &[Arc<SwapchainImage>],
        render_pass: Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> Vec<Arc<Framebuffer>> {
        let dimensions = images[0].dimensions().width_height();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(allocator, dimensions, Format::D16_UNORM).unwrap(),
        )
        .unwrap();

        let framebuffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view, depth_buffer.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        framebuffers
    }

    fn check_stage(&mut self, expected: RenderStage) -> bool {
        if self.render_stage == expected {
            return true;
        }

        match self.render_stage {
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                false
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                false
            }
        }
    }

    pub fn start(&mut self) {
        if !self.check_stage(RenderStage::Stopped) {
            return;
        }
        self.render_stage = RenderStage::Render;

        let (image_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain();
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            self.recreate_swapchain();
            return;
        }

        let clear_values = vec![Some([0.1, 0.7, 0.9, 1.0].into()), Some(1.0.into())];

        let mut commands = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        self.simulation.record(&mut commands);

        commands
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassContents::Inline,
            )
            .unwrap();

        commands
            .set_viewport(0, [self.viewport.clone()])
            .bind_pipeline_graphics(self.sky_pipeline.clone())
            .push_constants(self.sky_pipeline.layout().clone(), 0, self.sky_push)
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.sky_pipeline.layout().clone(),
                0,
                self.sky_set.clone(),
            )
            .draw(3, 1, 0, 0)
            .unwrap();

        self.commands = Some(commands);
        self.image_index = image_index;
        self.acquire_future = Some(acquire_future);
    }

    pub fn render(&mut self, draw_cache: &DrawCache) {
        if !self.check_stage(RenderStage::Render) {
            return;
        }

        let geometry_sets = draw_cache.geometry_sets.clone();
        let vertex_buffer = draw_cache.vertex_buffer.clone();
        let index_buffer = draw_cache.index_buffer.clone();
        let inst_buffer = draw_cache.inst_buffer.clone();
        self.commands
            .as_mut()
            .unwrap()
            .set_viewport(0, [self.viewport.clone()])
            .bind_pipeline_graphics(self.geometry_pipeline.clone())
            .push_constants(self.geometry_pipeline.layout().clone(), 0, self.camera_push)
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.geometry_pipeline.layout().clone(),
                0,
                geometry_sets,
            )
            .bind_vertex_buffers(0, (vertex_buffer.clone(), inst_buffer.clone()))
            .bind_index_buffer(index_buffer.clone())
            .draw_indexed(index_buffer.len() as u32, inst_buffer.len() as u32, 0, 0, 0)
            .unwrap();
    }

    pub fn finish(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
        if !self.check_stage(RenderStage::Render) {
            return;
        }

        let mut commands = self.commands.take().unwrap();
        commands.end_render_pass().unwrap();
        let command_buffer = commands.build().unwrap();

        let af = self.acquire_future.take().unwrap();

        let mut local_future: Option<Box<dyn GpuFuture>> =
            Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);

        mem::swap(&mut local_future, previous_frame_end);

        let future = local_future
            .take()
            .unwrap()
            .join(af)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    self.image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                *previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain();
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
        }

        self.commands = None;
        self.render_stage = RenderStage::Stopped;
    }
}
