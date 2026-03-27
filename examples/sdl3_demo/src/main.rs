//! SDL3 + wgpu: Cornell box → optional upscale (CLI `--upscale-amount`, output size) → swapchain.
//! Run: `cargo run -p sdl3_example -- --upscaler spatial` (see `--help`).

mod geometry;
mod scene;

use clap::{Parser, ValueEnum};
use scene::SceneRenderer;
use sdl3::event::Event;
use wgpu::util::TextureBlitter;
use wgpu::{CurrentSurfaceTexture, SurfaceTargetUnsafe};
use wgpu_scaler::{
    Upscaler, UpscalerDescriptor, UpscalerKind, temporal_antialiasing_jitter_pixels,
};

fn pixel_extent(window: &sdl3::video::Window) -> (u32, u32) {
    let (w, h) = window.size();
    let scale = window.display_scale();
    ((w as f32 * scale) as u32, (h as f32 * scale) as u32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum UpscalerCli {
    /// No upscaling; render at full resolution.
    None,
    MetalFXSpatial,
    MetalFXTemporal,
    Fsr2,
}

impl UpscalerCli {
    fn requested_kind(self) -> Option<UpscalerKind> {
        match self {
            UpscalerCli::None => None,
            UpscalerCli::MetalFXSpatial => Some(UpscalerKind::MetalFXSpatial),
            UpscalerCli::MetalFXTemporal => Some(UpscalerKind::MetalFXTemporal),
            UpscalerCli::Fsr2 => Some(UpscalerKind::FSR2),
        }
    }
}

#[derive(Parser)]
#[command(about = "Cornell box with selectable wgpu_scaler upscaler")]
struct Cli {
    #[arg(short, long, value_enum, default_value_t = UpscalerCli::None)]
    upscaler: UpscalerCli,

    /// Geometric upscale from render (input) to scaler output: each axis uses `input ≈ output / FACTOR`
    /// (default `2` ⇒ half-resolution scene, 2× upscale to output).
    #[arg(long, default_value_t = 2.0, value_name = "FACTOR")]
    upscale_amount: f32,
    /// Scaler output width in pixels (default: swapchain width).
    #[arg(long, value_name = "PX")]
    output_width: Option<u32>,
    /// Scaler output height in pixels (default: swapchain height).
    #[arg(long, value_name = "PX")]
    output_height: Option<u32>,
}

/// Resolves scaler I/O sizes from CLI and current drawable size (`>= 1` each).
fn scaler_sizes(cli: &Cli, surface_w: u32, surface_h: u32) -> (u32, u32, u32, u32, f32) {
    let out_w = cli.output_width.unwrap_or(surface_w).max(1);
    let out_h = cli.output_height.unwrap_or(surface_h).max(1);
    let factor = if cli.upscale_amount.is_finite() && cli.upscale_amount >= 1.0 {
        cli.upscale_amount
    } else {
        2.0
    };
    let in_w = ((out_w as f32 / factor).floor() as u32).max(1);
    let in_h = ((out_h as f32 / factor).floor() as u32).max(1);
    (in_w, in_h, out_w, out_h, factor)
}

/// Motion vectors, depth, Halton jitter, and storage-backed color textures for compute upscalers.
fn scaler_needs_temporal_scene_outputs(kind: UpscalerKind) -> bool {
    matches!(kind, UpscalerKind::MetalFXTemporal)
}

fn upscaler_active_label(kind: UpscalerKind) -> &'static str {
    match kind {
        UpscalerKind::MetalFXSpatial => "MetalFX spatial upscaler",
        UpscalerKind::MetalFXTemporal => {
            "MetalFX temporal upscaler / TAA denoiser (per-pixel motion from scene shader)"
        }
        UpscalerKind::FSR2 => "FSR2 (not implemented)",
    }
}

fn log_demo_state(
    note: &str,
    adapter: &wgpu::Adapter,
    cli: &Cli,
    logical_wh: (u32, u32),
    drawable_wh: (u32, u32),
    scale: f32,
    surface_format: wgpu::TextureFormat,
    resources: &Option<ScalerResources>,
    (in_w, in_h, out_w, out_h, upscale_factor): (u32, u32, u32, u32, f32),
) {
    let info = adapter.get_info();
    eprintln!("--- sdl3_example [{note}] ---");
    eprintln!("  adapter: {} | backend {:?}", info.name, info.backend);
    eprintln!(
        "  window logical: {}×{} px | drawable (swapchain): {}×{} px | content scale {:.2}×",
        logical_wh.0, logical_wh.1, drawable_wh.0, drawable_wh.1, scale
    );
    eprintln!("  surface format: {surface_format:?}");
    eprintln!("  CLI --upscaler: {:?}", cli.upscaler);
    eprintln!("  CLI --upscale-amount: {upscale_factor}× (input ≈ output / factor)");
    eprintln!("  scaler I/O (resolved): input {in_w}×{in_h} → output {out_w}×{out_h}",);
    if let Some(r) = resources {
        eprintln!("  active: {}", upscaler_active_label(r.kind));
    } else if cli.upscaler == UpscalerCli::None {
        eprintln!("  active: none (full-resolution scene → swapchain)");
    } else {
        eprintln!(
            "  active: none — {:?} could not be created on this device",
            cli.upscaler
        );
    }
    eprintln!("---");
}

#[allow(dead_code)]
struct ScalerResources {
    kind: UpscalerKind,
    scaler: Upscaler,
    color_in: wgpu::Texture,
    color_in_view: wgpu::TextureView,
    color_out: wgpu::Texture,
    color_out_view: wgpu::TextureView,
    motion: Option<(wgpu::Texture, wgpu::TextureView)>,
}

fn try_create_scaler_resources(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    kind: UpscalerKind,
) -> Option<ScalerResources> {
    let descriptor = UpscalerDescriptor {
        input_width: in_w,
        input_height: in_h,
        output_width: out_w,
        output_height: out_h,
        color_texture_format: format,
        motion_vectors_texture_format: wgpu::TextureFormat::Rg16Float,
        depth_texture_format: wgpu::TextureFormat::Depth32Float,
        output_texture_format: format,
    };
    let scaler = Upscaler::try_new(device, kind, descriptor)?;

    let motion = scaler_needs_temporal_scene_outputs(kind).then(|| {
        let motion_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("temporal motion"),
            size: wgpu::Extent3d {
                width: in_w,
                height: in_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let motion_view = motion_tex.create_view(&wgpu::TextureViewDescriptor::default());
        (motion_tex, motion_view)
    });

    // Temporal MetalFX runs compute that writes outputs; Metal requires shader-write usage on those
    // textures (`STORAGE_BINDING` → `MTLTextureUsageShaderWrite`).
    let color_in_usage = if scaler_needs_temporal_scene_outputs(kind) {
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
    } else {
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
    };
    let color_in = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scaler color in"),
        size: wgpu::Extent3d {
            width: in_w,
            height: in_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: color_in_usage,
        view_formats: &[],
    });
    let color_out_usage = if scaler_needs_temporal_scene_outputs(kind) {
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING
    } else {
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC
    };
    let color_out = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scaler color out"),
        size: wgpu::Extent3d {
            width: out_w,
            height: out_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: color_out_usage,
        view_formats: &[],
    });

    let color_in_view = color_in.create_view(&wgpu::TextureViewDescriptor::default());
    let color_out_view = color_out.create_view(&wgpu::TextureViewDescriptor::default());

    Some(ScalerResources {
        kind,
        scaler,
        color_in,
        color_in_view,
        color_out,
        color_out_view,
        motion,
    })
}

fn main() {
    let cli = Cli::parse();
    let requested_kind = cli.upscaler.requested_kind();

    let ctx = sdl3::init().unwrap();
    let window_title = match requested_kind {
        Some(k) => format!("wgpu_scaler — {k:?}"),
        None => "wgpu_scaler — no upscaler".to_string(),
    };
    let window = ctx
        .video()
        .unwrap()
        .window(&window_title, 1024, 768)
        .high_pixel_density()
        .build()
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::default(),
        flags: wgpu::InstanceFlags::default(),
        display: None,
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .unwrap();

    // - `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`: `Rg16Float` + `STORAGE_BINDING` for temporal MVs.
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu_scaler sdl3 example"),
        required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        ..Default::default()
    }))
    .unwrap();

    let surface = unsafe {
        instance
            .create_surface_unsafe(
                SurfaceTargetUnsafe::from_display_and_window(&window, &window).unwrap(),
            )
            .unwrap()
    };
    let cap = surface.get_capabilities(&adapter);
    let (width, height) = pixel_extent(&window);
    let surface_format = cap.formats[0];

    let mut surface_usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    surface_usage |= cap.usages & wgpu::TextureUsages::COPY_DST;

    let mut surface_config = wgpu::SurfaceConfiguration {
        usage: surface_usage,
        format: surface_format,
        view_formats: vec![surface_format.add_srgb_suffix()],
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        width,
        height,
        desired_maximum_frame_latency: 3,
        present_mode: wgpu::PresentMode::AutoVsync,
    };
    surface.configure(&device, &surface_config);

    let blitter = TextureBlitter::new(&device, surface_config.format);

    let (in_w, in_h, out_w, out_h, upscale_factor) = scaler_sizes(&cli, width, height);
    let mut resources = requested_kind.and_then(|kind| {
        try_create_scaler_resources(
            &device,
            surface_config.format,
            in_w,
            in_h,
            out_w,
            out_h,
            kind,
        )
    });

    let logical_wh = window.size();
    let scale = window.display_scale();
    log_demo_state(
        "startup",
        &adapter,
        &cli,
        logical_wh,
        (width, height),
        scale,
        surface_format,
        &resources,
        (in_w, in_h, out_w, out_h, upscale_factor),
    );

    let (scene_w, scene_h) = if resources.is_some() {
        (in_w, in_h)
    } else {
        (width, height)
    };
    let temporal_scene = resources
        .as_ref()
        .is_some_and(|r| scaler_needs_temporal_scene_outputs(r.kind));
    let mut scene = SceneRenderer::new(
        &device,
        surface_config.format,
        scene_w,
        scene_h,
        temporal_scene,
    );

    let mut event_pump = ctx.event_pump().unwrap();
    let mut running = true;
    let mut frame: u32 = 0;

    while running {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => running = false,
                Event::Window {
                    win_event: sdl3::event::WindowEvent::CloseRequested,
                    ..
                } => running = false,
                _ => {}
            }
        }

        let (nw, nh) = pixel_extent(&window);
        if nw != surface_config.width || nh != surface_config.height {
            surface_config.width = nw;
            surface_config.height = nh;
            surface.configure(&device, &surface_config);
            let (in_w, in_h, out_w, out_h, upscale_factor) = scaler_sizes(&cli, nw, nh);
            resources = requested_kind.and_then(|kind| {
                try_create_scaler_resources(
                    &device,
                    surface_config.format,
                    in_w,
                    in_h,
                    out_w,
                    out_h,
                    kind,
                )
            });
            let temporal_scene = resources
                .as_ref()
                .is_some_and(|r| scaler_needs_temporal_scene_outputs(r.kind));
            scene.set_color_format(&device, surface_config.format, temporal_scene);
            let logical_wh = window.size();
            let scale = window.display_scale();
            log_demo_state(
                "resize",
                &adapter,
                &cli,
                logical_wh,
                (nw, nh),
                scale,
                surface_config.format,
                &resources,
                (in_w, in_h, out_w, out_h, upscale_factor),
            );
        }

        let surface_tex = match surface.get_current_texture() {
            CurrentSurfaceTexture::Success(t) | CurrentSurfaceTexture::Suboptimal(t) => t,
            CurrentSurfaceTexture::Outdated => {
                surface.configure(&device, &surface_config);
                continue;
            }
            _ => continue,
        };

        let surface_view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let (tw, th) = if let Some(ref r) = resources {
            r.scaler.get_input_size()
        } else {
            (surface_config.width, surface_config.height)
        };
        scene.resize(&device, tw, th);
        let jitter_projection = resources
            .as_ref()
            .is_some_and(|r| scaler_needs_temporal_scene_outputs(r.kind));
        scene.update_camera(&queue, frame, jitter_projection);

        if let Some(ref mut r) = resources {
            let mut fill = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene low-res"),
            });

            let motion_view = scaler_needs_temporal_scene_outputs(r.kind).then(|| {
                r.motion
                    .as_ref()
                    .expect("temporal scaler has motion texture")
                    .1
                    .clone()
            });
            scene.render(&mut fill, &r.color_in_view, motion_view.as_ref());
            let fill_cb = fill.finish();

            r.scaler.set_color_texture(r.color_in_view.clone());
            r.scaler.set_output_texture(r.color_out_view.clone());

            if let Some(t) = r.scaler.as_temporal_mut() {
                t.set_motion_texture(
                    motion_view.expect("temporal scaler encodes motion in scene pass"),
                );
                t.set_depth_texture(scene.depth_view().clone());
                // Match Apple sample: Y flip for texture / Metal origin vs Halton pixel jitter.
                let (jx, jy) = temporal_antialiasing_jitter_pixels(frame);
                t.set_jitter_offset((jx, -jy));
                t.set_depth_reversed(false);
                t.set_motion_vector_scale((tw as f32, th as f32));
                t.set_reset(false);
            }

            let label = match r.kind {
                UpscalerKind::MetalFXSpatial => "MetalFX spatial",
                UpscalerKind::MetalFXTemporal => "MetalFX temporal",
                UpscalerKind::FSR2 => "FSR2 upscale",
            };
            let mut upscale_enc = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            r.scaler.dispatch(&mut upscale_enc);
            let upscale_cb = upscale_enc.finish();

            let mut present = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit to swapchain"),
            });
            blitter.copy(&device, &mut present, &r.color_out_view, &surface_view);
            let present_cb = present.finish();

            queue.submit([fill_cb, upscale_cb, present_cb]);
        } else {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene to swapchain"),
            });
            scene.render(&mut encoder, &surface_view, None);
            queue.submit(std::iter::once(encoder.finish()));
        }
        surface_tex.present();
        frame = frame.wrapping_add(1);
    }
}
