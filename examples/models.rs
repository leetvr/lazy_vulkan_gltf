use std::f32::consts::TAU;

use glam::Quat;
use lazy_vulkan::{BufferAllocation, LazyVulkan, Pipeline, SubRenderer, ash::vk};
use lazy_vulkan_gltf::LoadedAsset;
use winit::{
    application::ApplicationHandler, dpi::PhysicalSize, event::WindowEvent,
    window::WindowAttributes,
};

static VERTEX_SHADER_PATH: &'static str = "examples/shaders/main.vert.spv";
static FRAGMENT_SHADER_PATH: &'static str = "examples/shaders/main.frag.spv";

struct ModelRenderer {
    pipeline: Pipeline,
    index_buffer: BufferAllocation<u32>,
    models: Vec<Model>,
}

impl ModelRenderer {
    pub fn new(lazy_vulkan: &mut LazyVulkan) -> Self {
        let pipeline = lazy_vulkan
            .renderer
            .create_pipeline::<Registers>(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH);
        let mut index_buffer = lazy_vulkan
            .renderer
            .allocator
            .allocate_buffer(8192, vk::BufferUsageFlags::INDEX_BUFFER);

        let models = [
            ("test_assets/bullet.glb", glam::vec3(4.0, 0.0, 0.)),
            ("test_assets/cube.glb", glam::Vec3::ZERO),
        ]
        .into_iter()
        .map(|(path, position)| Model {
            asset: lazy_vulkan_gltf::load_asset(
                path,
                &mut lazy_vulkan.renderer.allocator,
                &mut lazy_vulkan.renderer.image_manager,
                &mut index_buffer,
            )
            .unwrap(),
            position,
        })
        .collect();

        ModelRenderer {
            pipeline,
            index_buffer,
            models,
        }
    }
}

impl SubRenderer for ModelRenderer {
    type State = RenderState;

    fn draw(
        &mut self,
        state: &Self::State,
        context: &lazy_vulkan::Context,
        params: lazy_vulkan::DrawParams,
    ) {
        self.begin_rendering(context, &self.pipeline);

        let device = &context.device;
        let command_buffer = context.draw_command_buffer;
        let mvp = build_mvp(state, params.drawable.extent);

        for model in &self.models {
            for mesh in &model.asset.meshes {
                for primitive in &mesh.primitives {
                    let registers = Registers {
                        mvp: mvp * glam::Affine3A::from_translation(model.position),
                        vertex_buffer: primitive.vertex_buffer.device_address,
                        material_buffer: primitive.material,
                    };
                    self.pipeline.update_registers(&registers);

                    unsafe {
                        device.cmd_bind_index_buffer(
                            command_buffer,
                            self.index_buffer.handle,
                            primitive.index_buffer_offset,
                            vk::IndexType::UINT32,
                        );
                        device.cmd_draw_indexed(command_buffer, primitive.index_count, 1, 0, 0, 0);
                    };
                }
            }
        }
    }

    fn stage_transfers(&mut self, _: &Self::State, _: &mut lazy_vulkan::Allocator) {}

    fn label(&self) -> &'static str {
        "Mesh Renderer"
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Registers {
    mvp: glam::Mat4,
    vertex_buffer: vk::DeviceAddress,
    material_buffer: vk::DeviceAddress,
}

unsafe impl bytemuck::Zeroable for Registers {}
unsafe impl bytemuck::Pod for Registers {}

struct Model {
    asset: LoadedAsset,
    position: glam::Vec3,
}

fn build_mvp(_state: &RenderState, extent: vk::Extent2D) -> glam::Mat4 {
    // Build up the perspective matrix
    let aspect_ratio = extent.width as f32 / extent.height as f32;
    let mut perspective =
        glam::Mat4::perspective_infinite_reverse_rh(60_f32.to_radians(), aspect_ratio, 0.01);

    // WULKAN
    perspective.y_axis *= -1.0;

    // Get view_from_world
    // TODO: camera
    let world_from_view = glam::Affine3A::from_rotation_translation(
        Quat::from_euler(glam::EulerRot::YXZ, TAU * 0.1, -TAU * 0.1, 0.),
        glam::Vec3::new(4., 4., 4.),
    );
    let view_from_world = world_from_view.inverse();

    perspective * view_from_world
}

// ============
// BOILERPLATE
// ============

#[derive(Default)]
struct RenderState {}

struct AppState {
    window: winit::window::Window,
    lazy_vulkan: LazyVulkan,
    sub_renderers: Vec<Box<dyn SubRenderer<State = RenderState>>>,
    render_state: RenderState,
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Models Example")
                    .with_inner_size(PhysicalSize::new(1024, 768)),
            )
            .unwrap();

        let mut lazy_vulkan = LazyVulkan::from_window(&window);

        let sub_renderers = vec![Box::new(ModelRenderer::new(&mut lazy_vulkan)) as _];

        self.state = Some(AppState {
            window,
            lazy_vulkan,
            sub_renderers,
            render_state: Default::default(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                state
                    .lazy_vulkan
                    .draw(&state.render_state, &mut state.sub_renderers);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        state.window.request_redraw();
    }
}

fn compile_shaders() {
    let _ = std::process::Command::new("glslc")
        .arg("examples/shaders/main.vert")
        .arg("-g")
        .arg("-o")
        .arg(VERTEX_SHADER_PATH)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();

    let _ = std::process::Command::new("glslc")
        .arg("examples/shaders/main.frag")
        .arg("-g")
        .arg("-o")
        .arg(FRAGMENT_SHADER_PATH)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
}

pub fn main() {
    use winit::platform::x11::EventLoopBuilderExtX11;

    env_logger::init();
    compile_shaders();
    let event_loop = winit::event_loop::EventLoopBuilder::default()
        .with_x11()
        .build()
        .unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
