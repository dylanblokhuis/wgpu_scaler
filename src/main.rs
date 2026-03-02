use sdl3::event::Event;
use wgpu::SurfaceTargetUnsafe;
use wgpu_scaler::{
    TemporalScaler, Upscaler, UpscalerDescriptor, UpscalerKind, supported_upscalers,
};

fn main() {
    let sdl_context = sdl3::init().unwrap();
    let _joystick_subsystem = sdl_context.joystick().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("wgpu_scaler", 1280, 720)
        .high_pixel_density()
        .vulkan()
        .build()
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        ..Default::default()
    }))
    .unwrap();

    let surface = unsafe {
        instance
            .create_surface_unsafe(SurfaceTargetUnsafe::from_window(&window).unwrap())
            .unwrap()
    };
    let cap = surface.get_capabilities(&adapter);
    let (width, height) = window.size();
    let content_scale = window.display_scale();
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: cap.formats[0],
        view_formats: vec![cap.formats[0].add_srgb_suffix()],
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        width: (width as f32 * content_scale) as u32,
        height: (height as f32 * content_scale) as u32,
        desired_maximum_frame_latency: 2,
        present_mode: wgpu::PresentMode::AutoVsync,
    };
    surface.configure(&device, &surface_config);

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut running = true;

    let supported = supported_upscalers(&adapter);
    let kind = supported
        .first()
        .copied()
        .unwrap_or(UpscalerKind::MetalFxTemporal);

    let mut scaler = Upscaler::new(
        &device,
        kind,
        UpscalerDescriptor {
            input_width: width,
            input_height: height,
            output_width: width,
            output_height: height,
            color_texture_format: wgpu::TextureFormat::Rgba8Unorm,
            motion_vectors_texture_format: wgpu::TextureFormat::Rg16Float,
            depth_texture_format: wgpu::TextureFormat::Depth32Float,
            output_texture_format: wgpu::TextureFormat::Rgba8UnormSrgb,
        },
    );

    while running {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => running = false,
                _ => (),
            }
        }

        let surface_texture = surface.get_current_texture().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

        scaler.dispatch(&mut encoder);

        queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }
}
