mod engine;
mod render;

use std::sync::{Arc, Mutex, RwLock};

use vulkano::sync;
use vulkano::sync::GpuFuture;

use winit::event::KeyboardInput;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::engine::{Camera, Engine, InputEvent, InputManager};

const PHYSICS_STEP_MS: u64 = 16;
const PHYSICS_STEP_SEC: f64 = PHYSICS_STEP_MS as f64 / 1000.0;

fn main() {
    // Just to make debug and release files work with debugger
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            if exe_dir.ends_with("debug") || exe_dir.ends_with("release") {
                let project_root = exe_dir.parent().unwrap().parent().unwrap();
                let _ = std::env::set_current_dir(project_root);
            }
        }
    }

    // Render setup
    let event_loop = EventLoop::new();
    let mut render = render::Render::new(&event_loop);

    // Engine setup
    let input_manager = Arc::new(Mutex::new(InputManager::new()));
    let camera = Arc::new(Mutex::new(Camera::new([0.0, 10.0, 0.0])));
    let engine = Arc::new(RwLock::new(Engine::new(
        input_manager.clone(),
        camera.clone(),
    )));

    // Initialize start state
    {
        let e = engine.read().unwrap();
        e.init();
    }

    // Physics thread
    {
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(PHYSICS_STEP_MS));
                let e = engine.write().unwrap();
                e.tick(PHYSICS_STEP_SEC);
            }
        });
    }

    // Render thread
    {
        let mut previous_frame_end =
            Some(Box::new(sync::now(render.device.clone())) as Box<dyn GpuFuture>);
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
                    let input_event = InputEvent::from_event_state(state, keycode);
                    input_manager.lock().unwrap().on_event(input_event);
                    //println!("Key event: {:?} {:?}", keycode, state);
                }

                WindowEvent::Focused(focused) => {
                    if focused {
                        render
                            .window()
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            .unwrap();
                    }
                }

                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }

                WindowEvent::Resized(_) => {
                    render.recreate_swapchain();
                }

                _ => {}
            },

            Event::DeviceEvent { event, .. } => match event {
                winit::event::DeviceEvent::MouseMotion { delta } => {
                    let input_event = InputEvent::from_mouse_motion(delta.0, delta.1);
                    input_manager.lock().unwrap().on_event(input_event);
                    //println!("Mouse moved: {:?} {:?}", delta.0, delta.1);
                }

                _ => {}
            },

            Event::RedrawEventsCleared => {
                render.window().request_redraw();
            }

            Event::RedrawRequested(_) => {
                previous_frame_end
                    .as_mut()
                    .take()
                    .unwrap()
                    .cleanup_finished();

                // Need to set scope this way so lock gets released before next frame
                {
                    let mut engine_camera = camera.lock().unwrap();
                    if engine_camera.is_changed {
                        engine_camera.update_matrices(render.aspect_ratio);
                        engine_camera.is_changed = !render.set_camera(&engine_camera);
                    }
                }

                render.start();
                render.voxel();
                render.finish(&mut previous_frame_end);
            }
            _ => (),
        });
    }
}
