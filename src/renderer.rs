use std::{mem, sync::Arc};

use vulkano::{
    VulkanLibrary,
    buffer::TypedBufferAccess,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{WriteDescriptorSet, allocator::StandardDescriptorSetAllocator},
    device::{
        self, Device, DeviceCreateInfo, Queue, QueueCreateInfo, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{AttachmentImage, ImageAccess, SwapchainImage, view::ImageView},
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
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    swapchain::{
        self, AcquireError, PresentMode, Surface, Swapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::window::{Window, WindowBuilder};

use crate::{
    camera::Camera,
    draw_cache::DrawCache,
    instance::{Instance, Mesh, Vertex},
    simulation::Simulation,
};

vulkano::impl_vertex!(Vertex, position, uv);
vulkano::impl_vertex!(Instance, instance_model, instance_normal);

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
    }
}

fn get_window(surface: &Arc<Surface>) -> &Window {
    surface.object().unwrap().downcast_ref::<Window>().unwrap()
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
    viewport: Viewport,
    framebuffers: Vec<Arc<Framebuffer>>,
    render_stage: RenderStage,
    commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    image_index: u32,
    acquire_future: Option<SwapchainAcquireFuture>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,

    pub texture_sampler: Arc<Sampler>,
    camera_push: water_vert::ty::Camera,
    pub simulation: Simulation,
}

impl Renderer {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        let instance = {
            let library = VulkanLibrary::new().unwrap();

            let mut extensions = vulkano_win::required_extensions(&library);
            extensions.khr_get_surface_capabilities2 = true;

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

        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
            .unwrap();
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

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
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

        let simulation = Simulation::new(
            &memory_allocator,
            &queue,
            &command_buffer_allocator,
            &device,
        );

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
            viewport,
            framebuffers,
            render_stage,
            commands,
            image_index,
            acquire_future,

            texture_sampler,
            camera_push,
            aspect_ratio,
            simulation,
        }
    }

    pub fn init(&mut self) {
        let mut commands = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        self.simulation.run_h0_spec(
            &mut commands,
            &self.descriptor_set_allocator,
            self.texture_sampler.clone(),
        );

        commands
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
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
        };
    }

    pub fn get_draw_cache(
        &self,
        mesh: &Mesh,
        instances: &Vec<Instance>,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
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

        let clear_values = vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())];

        let mut commands = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

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

        self.commands = Some(commands);
        self.image_index = image_index;
        self.acquire_future = Some(acquire_future);
    }

    pub fn render(&mut self, draw_cache: &DrawCache) {
        if !self.check_stage(RenderStage::Render) {
            return;
        }

        let geometry_set = draw_cache.geometry_set.clone();
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
                geometry_set,
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
