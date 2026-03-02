use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLPixelFormat;
use objc2_metal_fx::{MTLFXTemporalScaler, MTLFXTemporalScalerBase, MTLFXTemporalScalerDescriptor};
use sdl3::event::Event;
use wgpu::{AstcBlock, AstcChannel, SurfaceTargetUnsafe, hal::Adapter, wgc::api::Metal};

fn main() {
    let sdl_context = sdl3::init().unwrap();
    let joystick_subsystem = sdl_context.joystick().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window(&"wgpu_scaler", 1280, 720)
        .high_pixel_density()
        // .resizable()
        .vulkan()
        .build()
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        // backends: match backend.as_deref() {
        //     Some("vulkan") => wgpu::Backends::VULKAN,
        //     Some("metal") => wgpu::Backends::METAL,
        //     Some("dx12") => wgpu::Backends::DX12,
        //     Some(b) => {
        //         return Err(crate::dart_api::DartError::Api(format!(
        //             "Invalid GPU backend: {} - supported backends are: vulkan, metal, dx12",
        //             b
        //         )));
        //     }
        //     None => wgpu::Backends::VULKAN | wgpu::Backends::METAL,
        // },
        // flags: if cfg!(debug_assertions) {
        //     wgpu::InstanceFlags::default()
        // } else {
        //     wgpu::InstanceFlags::empty()
        // },
        // display: None,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        // required_features: wgpu::Features::IMMEDIATES
        //     | wgpu::Features::FLOAT32_FILTERABLE
        //     | wgpu::Features::BGRA8UNORM_STORAGE
        //     | wgpu::Features::RG11B10UFLOAT_RENDERABLE
        //     | wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS
        // | wgpu::Features::TEXTURE_BINDING_ARRAY
        // | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
        // | wgpu::Features::VERTEX_WRITABLE_STORAGE
        // | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        // | wgpu::Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
        // | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        // required_limits: wgpu::Limits {
        //     max_immediate_size: 128,
        //     max_storage_textures_per_shader_stage: 8,
        //     ..wgpu::Limits::default().using_minimum_supported_acceleration_structure_values()
        // },
        // experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
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

    let scaler = MetalTemporalScaler::new(
        &device,
        TemporalScalerDescriptor {
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

        let surface = surface.get_current_texture().unwrap();
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

        queue.submit(std::iter::once(encoder.finish()));
        surface.present();
    }
}

struct TemporalScalerDescriptor {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    color_texture_format: wgpu::TextureFormat,
    motion_vectors_texture_format: wgpu::TextureFormat,
    depth_texture_format: wgpu::TextureFormat,
    output_texture_format: wgpu::TextureFormat,
}

trait TemporalScaler {
    fn get_input_size(&self) -> (u32, u32);
    fn get_output_size(&self) -> (u32, u32);

    //
    fn set_jitter_offset(&mut self, jitter: (f32, f32));
    fn set_depth_reversed(&mut self, is_depth_reversed: bool);
    fn set_color_texture(&mut self, color_texture: wgpu::TextureView);
    fn set_motion_texture(&mut self, motion_texture: wgpu::TextureView);
    fn set_depth_texture(&mut self, depth_texture: wgpu::TextureView);
    fn set_output_texture(&mut self, output_texture: wgpu::TextureView);
    fn set_motion_vector_scale(&mut self, motion_vector_scale: (f32, f32));
    fn set_reset(&mut self, reset: bool);
    fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder);

    // preExposure?
    // exposureTexture?
}

struct MetalTemporalScaler {
    inner: Retained<ProtocolObject<dyn MTLFXTemporalScaler>>,
}

impl MetalTemporalScaler {
    fn new(device: &wgpu::Device, descriptor: TemporalScalerDescriptor) -> Self {
        let metal_device = unsafe { device.as_hal::<Metal>() }.unwrap();

        let temporal_scaler = unsafe {
            let desc = MTLFXTemporalScalerDescriptor::new();
            desc.setInputWidth(descriptor.input_width as _);
            desc.setInputHeight(descriptor.input_height as _);
            desc.setOutputWidth(descriptor.output_width as _);
            desc.setOutputHeight(descriptor.output_height as _);
            desc.setColorTextureFormat(map_format(descriptor.color_texture_format));
            desc.setMotionTextureFormat(map_format(descriptor.motion_vectors_texture_format));
            desc.setDepthTextureFormat(map_format(descriptor.depth_texture_format));
            desc.setOutputTextureFormat(map_format(descriptor.output_texture_format));

            let scaler = desc
                .newTemporalScalerWithDevice(metal_device.raw_device())
                .unwrap();

            scaler
        };

        MetalTemporalScaler {
            inner: temporal_scaler,
        }
    }
}

impl TemporalScaler for MetalTemporalScaler {
    fn get_input_size(&self) -> (u32, u32) {
        unsafe {
            (
                self.inner.inputWidth() as u32,
                self.inner.inputHeight() as u32,
            )
        }
    }
    fn get_output_size(&self) -> (u32, u32) {
        unsafe {
            (
                self.inner.outputWidth() as u32,
                self.inner.outputHeight() as u32,
            )
        }
    }
    fn set_reset(&mut self, reset: bool) {
        unsafe { self.inner.setReset(reset) };
    }
    fn set_color_texture(&mut self, color_texture: wgpu::TextureView) {
        unsafe {
            self.inner.setColorTexture(Some(
                &color_texture
                    .texture()
                    .as_hal::<Metal>()
                    .unwrap()
                    .raw_handle(),
            ))
        };
    }
    fn set_motion_texture(&mut self, motion_texture: wgpu::TextureView) {
        unsafe {
            self.inner.setMotionTexture(Some(
                &motion_texture
                    .texture()
                    .as_hal::<Metal>()
                    .unwrap()
                    .raw_handle(),
            ))
        };
    }
    fn set_depth_texture(&mut self, depth_texture: wgpu::TextureView) {
        unsafe {
            self.inner.setDepthTexture(Some(
                &depth_texture
                    .texture()
                    .as_hal::<Metal>()
                    .unwrap()
                    .raw_handle(),
            ))
        };
    }
    fn set_output_texture(&mut self, output_texture: wgpu::TextureView) {
        unsafe {
            self.inner.setOutputTexture(Some(
                &output_texture
                    .texture()
                    .as_hal::<Metal>()
                    .unwrap()
                    .raw_handle(),
            ))
        };
    }
    fn set_motion_vector_scale(&mut self, motion_vector_scale: (f32, f32)) {
        unsafe {
            self.inner.setMotionVectorScaleX(motion_vector_scale.0);
            self.inner.setMotionVectorScaleY(motion_vector_scale.1);
        }
    }
    fn set_jitter_offset(&mut self, jitter: (f32, f32)) {
        unsafe {
            self.inner.setJitterOffsetX(jitter.0);
            self.inner.setJitterOffsetY(jitter.1);
        }
    }
    fn set_depth_reversed(&mut self, is_depth_reversed: bool) {
        unsafe { self.inner.setDepthReversed(is_depth_reversed) };
    }

    fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder) {
        unsafe {
            encoder.as_hal_mut::<Metal, _, _>(|inner| {
                let handle = inner.unwrap().raw_command_buffer().unwrap();
                self.inner.encodeToCommandBuffer(&handle);
            });
        };
    }
}

fn map_format(format: wgpu::TextureFormat) -> MTLPixelFormat {
    use MTLPixelFormat as MTL;
    use wgpu::TextureFormat as Tf;
    match format {
        Tf::R8Unorm => MTL::R8Unorm,
        Tf::R8Snorm => MTL::R8Snorm,
        Tf::R8Uint => MTL::R8Uint,
        Tf::R8Sint => MTL::R8Sint,
        Tf::R16Uint => MTL::R16Uint,
        Tf::R16Sint => MTL::R16Sint,
        Tf::R16Unorm => MTL::R16Unorm,
        Tf::R16Snorm => MTL::R16Snorm,
        Tf::R16Float => MTL::R16Float,
        Tf::Rg8Unorm => MTL::RG8Unorm,
        Tf::Rg8Snorm => MTL::RG8Snorm,
        Tf::Rg8Uint => MTL::RG8Uint,
        Tf::Rg8Sint => MTL::RG8Sint,
        Tf::Rg16Unorm => MTL::RG16Unorm,
        Tf::Rg16Snorm => MTL::RG16Snorm,
        Tf::R32Uint => MTL::R32Uint,
        Tf::R32Sint => MTL::R32Sint,
        Tf::R32Float => MTL::R32Float,
        Tf::Rg16Uint => MTL::RG16Uint,
        Tf::Rg16Sint => MTL::RG16Sint,
        Tf::Rg16Float => MTL::RG16Float,
        Tf::Rgba8Unorm => MTL::RGBA8Unorm,
        Tf::Rgba8UnormSrgb => MTL::RGBA8Unorm_sRGB,
        Tf::Bgra8UnormSrgb => MTL::BGRA8Unorm_sRGB,
        Tf::Rgba8Snorm => MTL::RGBA8Snorm,
        Tf::Bgra8Unorm => MTL::BGRA8Unorm,
        Tf::Rgba8Uint => MTL::RGBA8Uint,
        Tf::Rgba8Sint => MTL::RGBA8Sint,
        Tf::Rgb10a2Uint => MTL::RGB10A2Uint,
        Tf::Rgb10a2Unorm => MTL::RGB10A2Unorm,
        Tf::Rg11b10Ufloat => MTL::RG11B10Float,
        // Ruint64 textures are emulated on metal
        Tf::R64Uint => MTL::RG32Uint,
        Tf::Rg32Uint => MTL::RG32Uint,
        Tf::Rg32Sint => MTL::RG32Sint,
        Tf::Rg32Float => MTL::RG32Float,
        Tf::Rgba16Uint => MTL::RGBA16Uint,
        Tf::Rgba16Sint => MTL::RGBA16Sint,
        Tf::Rgba16Unorm => MTL::RGBA16Unorm,
        Tf::Rgba16Snorm => MTL::RGBA16Snorm,
        Tf::Rgba16Float => MTL::RGBA16Float,
        Tf::Rgba32Uint => MTL::RGBA32Uint,
        Tf::Rgba32Sint => MTL::RGBA32Sint,
        Tf::Rgba32Float => MTL::RGBA32Float,
        Tf::Stencil8 => MTL::Stencil8,
        Tf::Depth16Unorm => MTL::Depth16Unorm,
        Tf::Depth32Float => MTL::Depth32Float,
        Tf::Depth32FloatStencil8 => MTL::Depth32Float_Stencil8,
        Tf::Depth24Plus => MTL::Depth24Unorm_Stencil8,
        Tf::Depth24PlusStencil8 => MTL::Depth32Float_Stencil8,
        Tf::NV12 => unreachable!(),
        Tf::P010 => unreachable!(),
        Tf::Rgb9e5Ufloat => MTL::RGB9E5Float,
        Tf::Bc1RgbaUnorm => MTL::BC1_RGBA,
        Tf::Bc1RgbaUnormSrgb => MTL::BC1_RGBA_sRGB,
        Tf::Bc2RgbaUnorm => MTL::BC2_RGBA,
        Tf::Bc2RgbaUnormSrgb => MTL::BC2_RGBA_sRGB,
        Tf::Bc3RgbaUnorm => MTL::BC3_RGBA,
        Tf::Bc3RgbaUnormSrgb => MTL::BC3_RGBA_sRGB,
        Tf::Bc4RUnorm => MTL::BC4_RUnorm,
        Tf::Bc4RSnorm => MTL::BC4_RSnorm,
        Tf::Bc5RgUnorm => MTL::BC5_RGUnorm,
        Tf::Bc5RgSnorm => MTL::BC5_RGSnorm,
        Tf::Bc6hRgbFloat => MTL::BC6H_RGBFloat,
        Tf::Bc6hRgbUfloat => MTL::BC6H_RGBUfloat,
        Tf::Bc7RgbaUnorm => MTL::BC7_RGBAUnorm,
        Tf::Bc7RgbaUnormSrgb => MTL::BC7_RGBAUnorm_sRGB,
        Tf::Etc2Rgb8Unorm => MTL::ETC2_RGB8,
        Tf::Etc2Rgb8UnormSrgb => MTL::ETC2_RGB8_sRGB,
        Tf::Etc2Rgb8A1Unorm => MTL::ETC2_RGB8A1,
        Tf::Etc2Rgb8A1UnormSrgb => MTL::ETC2_RGB8A1_sRGB,
        Tf::Etc2Rgba8Unorm => MTL::EAC_RGBA8,
        Tf::Etc2Rgba8UnormSrgb => MTL::EAC_RGBA8_sRGB,
        Tf::EacR11Unorm => MTL::EAC_R11Unorm,
        Tf::EacR11Snorm => MTL::EAC_R11Snorm,
        Tf::EacRg11Unorm => MTL::EAC_RG11Unorm,
        Tf::EacRg11Snorm => MTL::EAC_RG11Snorm,
        Tf::Astc { block, channel } => match channel {
            AstcChannel::Unorm => match block {
                AstcBlock::B4x4 => MTL::ASTC_4x4_LDR,
                AstcBlock::B5x4 => MTL::ASTC_5x4_LDR,
                AstcBlock::B5x5 => MTL::ASTC_5x5_LDR,
                AstcBlock::B6x5 => MTL::ASTC_6x5_LDR,
                AstcBlock::B6x6 => MTL::ASTC_6x6_LDR,
                AstcBlock::B8x5 => MTL::ASTC_8x5_LDR,
                AstcBlock::B8x6 => MTL::ASTC_8x6_LDR,
                AstcBlock::B8x8 => MTL::ASTC_8x8_LDR,
                AstcBlock::B10x5 => MTL::ASTC_10x5_LDR,
                AstcBlock::B10x6 => MTL::ASTC_10x6_LDR,
                AstcBlock::B10x8 => MTL::ASTC_10x8_LDR,
                AstcBlock::B10x10 => MTL::ASTC_10x10_LDR,
                AstcBlock::B12x10 => MTL::ASTC_12x10_LDR,
                AstcBlock::B12x12 => MTL::ASTC_12x12_LDR,
            },
            AstcChannel::UnormSrgb => match block {
                AstcBlock::B4x4 => MTL::ASTC_4x4_sRGB,
                AstcBlock::B5x4 => MTL::ASTC_5x4_sRGB,
                AstcBlock::B5x5 => MTL::ASTC_5x5_sRGB,
                AstcBlock::B6x5 => MTL::ASTC_6x5_sRGB,
                AstcBlock::B6x6 => MTL::ASTC_6x6_sRGB,
                AstcBlock::B8x5 => MTL::ASTC_8x5_sRGB,
                AstcBlock::B8x6 => MTL::ASTC_8x6_sRGB,
                AstcBlock::B8x8 => MTL::ASTC_8x8_sRGB,
                AstcBlock::B10x5 => MTL::ASTC_10x5_sRGB,
                AstcBlock::B10x6 => MTL::ASTC_10x6_sRGB,
                AstcBlock::B10x8 => MTL::ASTC_10x8_sRGB,
                AstcBlock::B10x10 => MTL::ASTC_10x10_sRGB,
                AstcBlock::B12x10 => MTL::ASTC_12x10_sRGB,
                AstcBlock::B12x12 => MTL::ASTC_12x12_sRGB,
            },
            AstcChannel::Hdr => match block {
                AstcBlock::B4x4 => MTL::ASTC_4x4_HDR,
                AstcBlock::B5x4 => MTL::ASTC_5x4_HDR,
                AstcBlock::B5x5 => MTL::ASTC_5x5_HDR,
                AstcBlock::B6x5 => MTL::ASTC_6x5_HDR,
                AstcBlock::B6x6 => MTL::ASTC_6x6_HDR,
                AstcBlock::B8x5 => MTL::ASTC_8x5_HDR,
                AstcBlock::B8x6 => MTL::ASTC_8x6_HDR,
                AstcBlock::B8x8 => MTL::ASTC_8x8_HDR,
                AstcBlock::B10x5 => MTL::ASTC_10x5_HDR,
                AstcBlock::B10x6 => MTL::ASTC_10x6_HDR,
                AstcBlock::B10x8 => MTL::ASTC_10x8_HDR,
                AstcBlock::B10x10 => MTL::ASTC_10x10_HDR,
                AstcBlock::B12x10 => MTL::ASTC_12x10_HDR,
                AstcBlock::B12x12 => MTL::ASTC_12x12_HDR,
            },
        },
    }
}

fn halton(index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut result = 0.0;
    let mut i = index;

    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i /= base;
    }
    result
}

fn jitter_phase_count(render_width: i32, display_width: i32) -> i32 {
    let base_phase_count = 8.0_f32;
    let scale_factor = display_width as f32 / render_width as f32;
    let jitter_phase_count = base_phase_count * scale_factor.powi(2);

    jitter_phase_count as i32
}

pub fn get_generic_jitter_offset(
    input_width: i32,
    output_width: i32,
    frame_index: u32,
) -> (f32, f32) {
    let phase_count = jitter_phase_count(input_width, output_width);
    let jitter_offset_x = halton(frame_index, phase_count as u32);
    let jitter_offset_y = halton(frame_index, phase_count as u32);
    (jitter_offset_x, jitter_offset_y)
}
