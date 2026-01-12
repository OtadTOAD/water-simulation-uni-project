pub mod dummy_vertex;

use crate::engine::Camera;
use dummy_vertex::DummyVertex;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, SwapchainImage};
use vulkano::instance::debug::{
    DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
    DebugUtilsMessengerCreateInfo,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;

use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    self, AcquireError, PresentMode, Surface, Swapchain, SwapchainAcquireFuture,
    SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::{Version, VulkanLibrary};

use vulkano_win::VkSurfaceBuild;

use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use std::mem;
use std::sync::Arc;

vulkano::impl_vertex!(DummyVertex, position);

mod water_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/render/shaders/water.vert",
    }
}

mod water_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/render/shaders/water.frag",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

#[derive(Debug, Clone)]
enum RenderStage {
    Stopped,
    Voxel,
    NeedsRedraw,
}

pub struct Render {
    pub device: Arc<Device>,
    pub aspect_ratio: f32,

    surface: Arc<Surface>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    render_pass: Arc<RenderPass>,
    voxel_pipeline: Arc<GraphicsPipeline>,
    dummy_verts: Arc<CpuAccessibleBuffer<[DummyVertex]>>,
    viewport: Viewport,
    framebuffers: Vec<Arc<Framebuffer>>,
    render_stage: RenderStage,
    commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    image_index: u32,
    acquire_future: Option<SwapchainAcquireFuture>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,

    camera_buffer: Arc<CpuAccessibleBuffer<water_frag::ty::Camera>>,
}

impl Render {
    pub fn new(event_loop: &EventLoop<()>) -> Render {
        let instance = {
            let library = VulkanLibrary::new().unwrap();

            let mut extensions = vulkano_win::required_extensions(&library);
            extensions.khr_get_surface_capabilities2 = false;

            let mut layers = vec![];
            if library
                .layer_properties()
                .unwrap()
                .into_iter()
                .any(|l| l.name() == "VK_LAYER_KHRONOS_validation")
            {
                layers.push("VK_LAYER_KHRONOS_validation".to_string());
            } else {
                println!("NO VALIDATION!")
            }

            Instance::new(
                library,
                InstanceCreateInfo {
                    enabled_extensions: extensions,
                    enumerate_portability: true,
                    max_api_version: Some(Version::V1_1),
                    enabled_layers: layers,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        unsafe {
            let mut severity = DebugUtilsMessageSeverity::empty();
            severity.error = true;
            severity.verbose = true;
            severity.warning = true;
            severity.information = true;
            let mut debug_type = DebugUtilsMessageType::empty();
            debug_type.validation = true;
            debug_type.performance = true;
            debug_type.general = true;

            let _debug_messenger = DebugUtilsMessenger::new(
                instance.clone(),
                DebugUtilsMessengerCreateInfo {
                    message_severity: severity,
                    message_type: debug_type,
                    ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                        println!("[VULKAN {:?}] {}", msg.severity, msg.description);
                    }))
                },
            )
            .ok();
        }

        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ext_full_screen_exclusive: false,
            ..DeviceExtensions::empty()
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

            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
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

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let deferred_vert = water_vert::load(device.clone()).unwrap();
        let deferred_frag = water_frag::load(device.clone()).unwrap();

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

        let voxel_pass = Subpass::from(render_pass.clone(), 0).unwrap();

        let voxel_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<DummyVertex>())
            .vertex_shader(deferred_vert.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(deferred_frag.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::disabled())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .render_pass(voxel_pass.clone())
            .build(device.clone())
            .unwrap();

        let dummy_verts = CpuAccessibleBuffer::from_iter(
            &memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            DummyVertex::list().iter().cloned(),
        )
        .unwrap();

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        let framebuffers = Render::window_size_dependent_setup(
            &memory_allocator,
            &images,
            render_pass.clone(),
            &mut viewport,
        );

        let camera_buffer = CpuAccessibleBuffer::from_data(
            &memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            water_frag::ty::Camera {
                invProj: [[0.0; 4]; 4],
                invView: [[0.0; 4]; 4],
                camPos: [0.0; 3],
            },
        )
        .unwrap();

        let render_stage = RenderStage::Stopped;
        let commands = None;
        let image_index = 0;
        let acquire_future = None;

        let aspect_ratio = {
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            window.inner_size().width as f32 / window.inner_size().height as f32
        };

        Render {
            surface,
            device,
            queue,
            swapchain,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            render_pass,
            voxel_pipeline,
            dummy_verts,
            viewport,
            framebuffers,
            render_stage,
            commands,
            image_index,
            acquire_future,

            camera_buffer,
            aspect_ratio,
        }
    }

    pub fn window(&self) -> &Window {
        self.surface
            .object()
            .unwrap()
            .downcast_ref::<Window>()
            .unwrap()
    }

    pub fn set_camera(&mut self, camera: &Camera) -> bool {
        match self.camera_buffer.write() {
            Ok(mut content) => {
                let (inv_proj, inv_view) = camera.get_matrices();
                content.camPos = camera.position;
                content.invProj = *inv_proj;
                content.invView = *inv_view;

                return true;
            }
            Err(_) => return false,
        }
    }

    pub fn voxel(&mut self) {
        match self.render_stage {
            RenderStage::Voxel => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let voxel_layout = self.voxel_pipeline.layout().set_layouts().get(0).unwrap();
        let voxel_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            voxel_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.camera_buffer.clone())],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .set_viewport(0, [self.viewport.clone()])
            .bind_pipeline_graphics(self.voxel_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.voxel_pipeline.layout().clone(),
                0,
                voxel_set.clone(),
            )
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();
    }

    pub fn finish(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
        match self.render_stage {
            RenderStage::Voxel => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
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

    pub fn start(&mut self) {
        match self.render_stage {
            RenderStage::Stopped => {
                self.render_stage = RenderStage::Voxel;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

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

        let new_framebuffers = Render::window_size_dependent_setup(
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
}
