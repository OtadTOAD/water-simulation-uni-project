mod camera;
mod instance;
mod renderer;

use nalgebra_glm::Vec3;
use vulkano::sync::{self, GpuFuture};
use winit::{
    event::{Event, KeyboardInput, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::{camera::Camera, renderer::Renderer};

fn main() {
    let event_loop = EventLoop::new();
    let mut renderer = Renderer::new(&event_loop);

    let camera = Camera::new(Vec3::new(-10.0, -1.0, 0.0));

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
            } => {
                println!("Key event: {:?} {:?}", keycode, state);
            }

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
                println!("Mouse moved: {:?} {:?}", delta.0, delta.1);
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

            renderer.start();
            renderer.render();
            renderer.finish(&mut previous_frame_end);
        }
        _ => (),
    });
}
