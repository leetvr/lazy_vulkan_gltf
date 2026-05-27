use std::f32::consts::TAU;

use glam::Quat;
use lazy_vulkan::{BufferAllocation, LazyVulkan, Pipeline, StateFamily, SubRenderer, ash::vk};
use lazy_vulkan_gltf::{LoadedAsset, Vertex};
use winit::{
    application::ApplicationHandler, dpi::PhysicalSize, event::WindowEvent,
    window::WindowAttributes,
};

static VERTEX_SHADER_SOURCE_PATH: &str = "examples/shaders/main.vert.slang";
static FRAGMENT_SHADER_SOURCE_PATH: &str = "examples/shaders/main.frag.slang";
static VERTEX_SHADER_PATH: &str = "examples/shaders/main.vert.spv";
static FRAGMENT_SHADER_PATH: &str = "examples/shaders/main.frag.spv";

struct RenderStateFamily;

impl StateFamily for RenderStateFamily {
    type For<'s> = RenderState;
}

struct RenderState {
    extent: vk::Extent2D,
}

struct ModelRenderer {
    pipeline: Pipeline,
    vertex_buffer: BufferAllocation<Vertex>,
    index_buffer: BufferAllocation<u32>,
    models: Vec<Model>,
}

impl ModelRenderer {
    pub fn new(renderer: &mut lazy_vulkan::Renderer<RenderStateFamily>) -> Self {
        let pipeline =
            renderer.create_pipeline::<Registers>(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH);
        let mut vertex_buffer = renderer
            .allocator
            .allocate_buffer(10 * 1024 * 1024, vk::BufferUsageFlags::STORAGE_BUFFER);

        let mut index_buffer = renderer
            .allocator
            .allocate_buffer(1024 * 1024, vk::BufferUsageFlags::INDEX_BUFFER);

        let models = [
            ("test_assets/bullet.glb", glam::vec3(4.0, 0.0, 0.)),
            ("test_assets/cube.glb", glam::Vec3::ZERO),
        ]
        .into_iter()
        .map(|(path, position)| Model {
            asset: lazy_vulkan_gltf::load_asset(
                path,
                &mut renderer.allocator,
                &mut renderer.image_manager,
                &mut index_buffer,
                &mut vertex_buffer,
            )
            .unwrap(),
            position,
        })
        .collect();

        ModelRenderer {
            pipeline,
            vertex_buffer,
            index_buffer,
            models,
        }
    }
}

impl SubRenderer<'_> for ModelRenderer {
    type State = RenderState;

    fn draw_opaque(&mut self, state: &Self::State, context: &lazy_vulkan::Context) {
        self.begin_rendering(context, &self.pipeline);

        let device = &context.device;
        let command_buffer = context.draw_command_buffer;
        let mvp = build_mvp(state.extent);

        for model in &self.models {
            for node in &model.asset.nodes {
                let mesh = &model.asset.meshes[usize::from(node.mesh_id)];
                for primitive in &mesh.primitives {
                    let vertex_buffer =
                        self.vertex_buffer.device_address + primitive.vertex_buffer_offset;

                    let registers = Registers {
                        mvp: mvp
                            * glam::Affine3A::from_rotation_translation(
                                glam::Quat::IDENTITY,
                                model.position,
                            )
                            * node.transform,
                        vertex_buffer,
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

fn build_mvp(extent: vk::Extent2D) -> glam::Mat4 {
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

struct AppState {
    window: winit::window::Window,
    lazy_vulkan: LazyVulkan<RenderStateFamily>,
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

        let model_renderer = ModelRenderer::new(&mut lazy_vulkan.renderer);
        lazy_vulkan.add_sub_renderer(Box::new(model_renderer));

        self.state = Some(AppState {
            window,
            lazy_vulkan,
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
                state.window.pre_present_notify();
                state.lazy_vulkan.draw(&RenderState {
                    extent: vk::Extent2D {
                        width: state.window.inner_size().width,
                        height: state.window.inner_size().height,
                    },
                });
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn compile_shaders() {
    use shader_slang as slang;

    let global_session = slang::GlobalSession::new().unwrap();
    let shader_search_path = std::ffi::CString::new("examples/shaders/").unwrap();

    let session_options = slang::CompilerOptions::default()
        .language(slang::SourceLanguage::Slang)
        .optimization(slang::OptimizationLevel::None)
        .debug_information(slang::DebugInfoLevel::Standard)
        .glsl_force_scalar_layout(true)
        .matrix_layout_row(true)
        .vulkan_use_entry_point_name(true)
        .emit_spirv_directly(true);

    let target_desc = slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global_session.find_profile("spirv_1_6"));

    let targets = [target_desc];
    let search_paths = [shader_search_path.as_ptr()];
    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = global_session.create_session(&session_desc).unwrap();

    compile_shader(&session, VERTEX_SHADER_SOURCE_PATH, VERTEX_SHADER_PATH);
    compile_shader(&session, FRAGMENT_SHADER_SOURCE_PATH, FRAGMENT_SHADER_PATH);
}

fn compile_shader(session: &shader_slang::Session, source_path: &str, output_path: &str) {
    let module = session
        .load_module(source_path)
        .unwrap_or_else(|err| panic!("failed to load {source_path}: {err}"));
    let entry_point = module
        .find_entry_point_by_name("main")
        .unwrap_or_else(|| panic!("failed to find main in {source_path}"));
    let program = session
        .create_composite_component_type(&[module.clone().into(), entry_point.into()])
        .unwrap_or_else(|err| panic!("failed to create Slang program for {source_path}: {err}"));
    let linked_program = program
        .link()
        .unwrap_or_else(|err| panic!("failed to link {source_path}: {err}"));
    let shader_bytecode = linked_program
        .entry_point_code(0, 0)
        .unwrap_or_else(|err| panic!("failed to emit SPIR-V for {source_path}: {err}"));

    std::fs::write(output_path, shader_bytecode.as_slice())
        .unwrap_or_else(|err| panic!("failed to write {output_path}: {err}"));
}

#[cfg(test)]
mod tests {
    #[test]
    fn compiles_slang_shaders() {
        super::compile_shaders();
    }
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
