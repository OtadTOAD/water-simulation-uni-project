mod camera;
mod draw_cache;
mod instance;
mod renderer;
mod simulation;
mod water;

use nalgebra_glm::{IVec3, Vec3};
use vulkano::{
    descriptor_set::WriteDescriptorSet,
    sync::{self, GpuFuture},
};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::{camera::Camera, renderer::Renderer, water::Water};

fn main() {
    let event_loop = EventLoop::new();
    let mut renderer = Renderer::new(&event_loop);

    let mut camera = Camera::new(Vec3::new(-10.0, -1.0, 0.0));
    let mut move_dir = IVec3::new(0, 0, 0);

    let simulation = renderer.get_simulation();

    let water = Water::new();
    let water_cache = renderer.get_draw_cache(
        &water.mesh,
        &water.instances,
        [WriteDescriptorSet::image_view_sampler(
            0,
            simulation.noise_image.clone(),
            renderer.texture_sampler.clone(),
        )],
    );

    let mut previous_frame_end =
        Some(Box::new(sync::now(renderer.device.clone())) as Box<dyn GpuFuture>);
    let mut last_frame_time = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => match (keycode, state) {
                (VirtualKeyCode::Escape, _) => {
                    *control_flow = ControlFlow::Exit;
                }
                (VirtualKeyCode::W, x) => {
                    if x == ElementState::Pressed {
                        move_dir.y = 1;
                    } else {
                        move_dir.y = 0;
                    }
                }
                (VirtualKeyCode::S, x) => {
                    if x == ElementState::Pressed {
                        move_dir.y = -1;
                    } else {
                        move_dir.y = 0;
                    }
                }
                (VirtualKeyCode::A, x) => {
                    if x == ElementState::Pressed {
                        move_dir.x = -1;
                    } else {
                        move_dir.x = 0;
                    }
                }
                (VirtualKeyCode::D, x) => {
                    if x == ElementState::Pressed {
                        move_dir.x = 1;
                    } else {
                        move_dir.x = 0;
                    }
                }
                (VirtualKeyCode::Space, x) => {
                    if x == ElementState::Pressed {
                        move_dir.z = 1;
                    } else {
                        move_dir.z = 0;
                    }
                }
                (VirtualKeyCode::LShift, x) => {
                    if x == ElementState::Pressed {
                        move_dir.z = -1;
                    } else {
                        move_dir.z = 0;
                    }
                }
                _ => {}
            },

            WindowEvent::Focused(focused) => {
                if focused {
                    renderer
                        .window()
                        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                        .unwrap();
                }
            }

            WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }

            WindowEvent::Resized(_) => {
                renderer.recreate_swapchain();
            }

            _ => {}
        },

        Event::DeviceEvent { event, .. } => match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                camera.on_mouse_dlta(delta.0 as f32, delta.1 as f32);
            }

            _ => {}
        },

        Event::RedrawEventsCleared => {
            renderer.window().request_redraw();
        }

        Event::RedrawRequested(_) => {
            let curr_time = std::time::Instant::now();
            let delta_time = curr_time.duration_since(last_frame_time).as_secs_f32();
            last_frame_time = curr_time;

            println!("Frame Rate: {:.2}", 1.0 / delta_time);

            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            let updated = camera.tick(&move_dir, delta_time, renderer.aspect_ratio);
            if updated {
                renderer.set_camera(&camera);
            }

            renderer.start();
            renderer.render(&water_cache);
            renderer.finish(&mut previous_frame_end);
        }
        _ => (),
    });
}
