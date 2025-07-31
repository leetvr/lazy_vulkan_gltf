use lazy_vulkan::{BufferAllocation, LazyVulkan, Pipeline, SubRenderer, ash::vk};
use lazy_vulkan_gltf::LoadedAsset;
use winit::{
    application::ApplicationHandler, dpi::PhysicalSize, event::WindowEvent,
    window::WindowAttributes,
};

#[repr(C)]
#[derive(Copy, Clone)]
struct Registers {}

unsafe impl bytemuck::Zeroable for Registers {}
unsafe impl bytemuck::Pod for Registers {}

struct ModelRenderer {
    pipeline: Pipeline,
    index_buffer: BufferAllocation<u32>,
    asset: LoadedAsset,
}

impl ModelRenderer {
    pub fn new(lazy_vulkan: &mut LazyVulkan) -> Self {
        let pipeline = lazy_vulkan
            .renderer
            .create_pipeline::<Registers>("shaders/vert.spv", "shaders/frag.spv");
        let mut index_buffer = lazy_vulkan
            .renderer
            .allocator
            .allocate_buffer(100_000, vk::BufferUsageFlags::STORAGE_BUFFER);

        let asset = lazy_vulkan_gltf::load_asset(
            "test_assets/bullet.glb",
            &mut lazy_vulkan.renderer.allocator,
            &mut lazy_vulkan.renderer.image_manager,
            &mut index_buffer,
        )
        .unwrap();

        ModelRenderer {
            pipeline,
            index_buffer,
            asset,
        }
    }
}

impl SubRenderer for ModelRenderer {
    type State = RenderState;

    fn draw(
        &mut self,
        _state: &Self::State,
        context: &lazy_vulkan::Context,
        _params: lazy_vulkan::DrawParams,
    ) {
        self.begin_rendering(context, &self.pipeline);

        for model in &self.asset.meshes {
            for primitive in &model.primitives {
                let registers = Registers {};
                let device = &context.device;
                let command_buffer = context.draw_command_buffer;
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

    fn stage_transfers(&mut self, _: &Self::State, _: &mut lazy_vulkan::Allocator) {
        todo!()
    }
}

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

pub fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
