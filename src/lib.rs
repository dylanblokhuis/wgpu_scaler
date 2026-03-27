//! wgpu_scaler: cross-backend upscaling (MetalFX temporal and spatial, and more).

#[cfg(target_os = "macos")]
mod metalfx;

/// Identifies an upscaler implementation (MetalFX temporal, spatial, FSR, NIS, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpscalerKind {
    /// MetalFX temporal scaler (Metal backend only).
    MetalFXTemporal,
    /// MetalFX spatial scaler (Metal backend only).
    MetalFXSpatial,
    FSR2,
}

/// Descriptor for a MetalFX temporal scaler (color, depth, motion, and output formats).
#[derive(Debug, Clone)]
pub struct TemporalUpscalerDescriptor {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub color_texture_format: wgpu::TextureFormat,
    pub motion_vectors_texture_format: wgpu::TextureFormat,
    pub depth_texture_format: wgpu::TextureFormat,
    pub output_texture_format: wgpu::TextureFormat,
    /// If `true`, MetalFX computes exposure each frame and ignores [`TemporalScaler::set_exposure_texture`].
    pub auto_exposure: bool,
    /// When `true`, MetalFX compiles the upscaler up front (slower creation, faster first frames).
    pub requires_synchronous_initialization: bool,
    /// Per-pixel reactive mask (see [`TemporalScaler::set_reactive_mask_texture`]).
    pub reactive_mask_enabled: bool,
    pub reactive_mask_texture_format: wgpu::TextureFormat,
    /// Dynamic resolution: interpret `input_content_min_scale` / `input_content_max_scale` on the scaler.
    pub input_content_properties_enabled: bool,
    pub input_content_min_scale: f32,
    pub input_content_max_scale: f32,
}

impl Default for TemporalUpscalerDescriptor {
    fn default() -> Self {
        Self {
            input_width: 0,
            input_height: 0,
            output_width: 0,
            output_height: 0,
            color_texture_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            motion_vectors_texture_format: wgpu::TextureFormat::Rg16Float,
            depth_texture_format: wgpu::TextureFormat::Depth32Float,
            output_texture_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            auto_exposure: false,
            requires_synchronous_initialization: false,
            reactive_mask_enabled: false,
            reactive_mask_texture_format: wgpu::TextureFormat::R8Unorm,
            input_content_properties_enabled: false,
            input_content_min_scale: 1.0,
            input_content_max_scale: 1.0,
        }
    }
}

/// Descriptor for a MetalFX spatial scaler (input color and output only).
#[derive(Debug, Clone)]
pub struct SpatialUpscalerDescriptor {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub color_texture_format: wgpu::TextureFormat,
    pub output_texture_format: wgpu::TextureFormat,
}

/// Legacy combined descriptor: use [`TemporalUpscalerDescriptor`] or [`SpatialUpscalerDescriptor`] for new code.
#[derive(Debug, Clone)]
pub struct UpscalerDescriptor {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub color_texture_format: wgpu::TextureFormat,
    pub motion_vectors_texture_format: wgpu::TextureFormat,
    pub depth_texture_format: wgpu::TextureFormat,
    pub output_texture_format: wgpu::TextureFormat,
}

impl From<UpscalerDescriptor> for TemporalUpscalerDescriptor {
    fn from(d: UpscalerDescriptor) -> Self {
        Self {
            input_width: d.input_width,
            input_height: d.input_height,
            output_width: d.output_width,
            output_height: d.output_height,
            color_texture_format: d.color_texture_format,
            motion_vectors_texture_format: d.motion_vectors_texture_format,
            depth_texture_format: d.depth_texture_format,
            output_texture_format: d.output_texture_format,
            auto_exposure: false,
            requires_synchronous_initialization: false,
            reactive_mask_enabled: false,
            reactive_mask_texture_format: wgpu::TextureFormat::R8Unorm,
            input_content_properties_enabled: false,
            input_content_min_scale: 1.0,
            input_content_max_scale: 1.0,
        }
    }
}

impl From<UpscalerDescriptor> for SpatialUpscalerDescriptor {
    fn from(d: UpscalerDescriptor) -> Self {
        Self {
            input_width: d.input_width,
            input_height: d.input_height,
            output_width: d.output_width,
            output_height: d.output_height,
            color_texture_format: d.color_texture_format,
            output_texture_format: d.output_texture_format,
        }
    }
}

/// Returns which upscalers are supported for the given adapter (e.g. by backend).
pub fn supported_upscalers(adapter: &wgpu::Adapter) -> Vec<UpscalerKind> {
    let info = adapter.get_info();
    let mut out = Vec::new();
    match info.backend {
        wgpu::Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                let _ = adapter;
                // MetalFX availability is per `MTLDevice`; wgpu does not expose the adapter’s
                // raw device until you have a `Device`. Use [`Upscaler::try_new`] to confirm.
                out.push(UpscalerKind::MetalFXTemporal);
                out.push(UpscalerKind::MetalFXSpatial);
            }
        }
        wgpu::Backend::Vulkan => {}
        _ => {}
    }
    out
}

/// MetalFX spatial scaler interface (input color → upscaled output).
pub trait SpatialScaler {
    fn get_input_size(&self) -> (u32, u32);
    fn get_output_size(&self) -> (u32, u32);
    fn set_color_texture(&mut self, color_texture: wgpu::TextureView);
    fn set_output_texture(&mut self, output_texture: wgpu::TextureView);
    /// Active region inside the color texture (defaults to full [`Self::get_input_size`] when equal).
    fn set_input_content_size(&mut self, width: u32, height: u32);
    /// On Metal, this uses the raw HAL encoder; do not mix with other wgpu recording on the same
    /// [`wgpu::CommandEncoder`] (use a dedicated encoder, then [`wgpu::Queue::submit`] in order).
    fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder);
}

/// MetalFX temporal scaler interface (color + depth + motion → upscaled output).
pub trait TemporalScaler {
    fn get_input_size(&self) -> (u32, u32);
    fn get_output_size(&self) -> (u32, u32);
    fn set_jitter_offset(&mut self, jitter: (f32, f32));
    fn set_depth_reversed(&mut self, is_depth_reversed: bool);
    fn set_color_texture(&mut self, color_texture: wgpu::TextureView);
    fn set_motion_texture(&mut self, motion_texture: wgpu::TextureView);
    fn set_depth_texture(&mut self, depth_texture: wgpu::TextureView);
    fn set_output_texture(&mut self, output_texture: wgpu::TextureView);
    fn set_motion_vector_scale(&mut self, motion_vector_scale: (f32, f32));
    fn set_reset(&mut self, reset: bool);
    /// On Metal, this uses the raw HAL encoder; do not mix with other wgpu recording on the same
    /// [`wgpu::CommandEncoder`] (use a dedicated encoder, then [`wgpu::Queue::submit`] in order).
    fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder);

    fn set_input_content_size(&mut self, width: u32, height: u32) {
        let _ = (width, height);
    }

    /// 1×1 `R16Float` exposure texture; ignored when the scaler was created with `auto_exposure: true`.
    fn set_exposure_texture(&mut self, exposure_texture: Option<wgpu::TextureView>) {
        let _ = exposure_texture;
    }

    fn set_reactive_mask_texture(&mut self, mask: Option<wgpu::TextureView>) {
        let _ = mask;
    }

    fn set_pre_exposure(&mut self, pre_exposure: f32) {
        let _ = pre_exposure;
    }
}

/// Holds either a spatial or temporal MetalFX scaler.
pub enum Upscaler {
    Spatial(Box<dyn SpatialScaler>),
    Temporal(Box<dyn TemporalScaler>),
}

impl Upscaler {
    /// Create a scaler of `kind`, or `None` if creation fails (unsupported GPU or dimensions).
    pub fn try_new(
        device: &wgpu::Device,
        kind: UpscalerKind,
        descriptor: UpscalerDescriptor,
    ) -> Option<Self> {
        let temporal: TemporalUpscalerDescriptor = descriptor.clone().into();
        let spatial: SpatialUpscalerDescriptor = descriptor.into();
        Self::try_new_from_parts(device, kind, temporal, spatial)
    }

    /// Create with explicit spatial and temporal descriptors (only the one matching `kind` is used).
    pub fn try_new_from_parts(
        device: &wgpu::Device,
        kind: UpscalerKind,
        temporal: TemporalUpscalerDescriptor,
        spatial: SpatialUpscalerDescriptor,
    ) -> Option<Self> {
        match kind {
            UpscalerKind::MetalFXTemporal => {
                #[cfg(target_os = "macos")]
                {
                    metalfx::MetalTemporalScaler::try_new(device, temporal)
                        .map(|s| Upscaler::Temporal(Box::new(s)))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (device, temporal);
                    None
                }
            }
            UpscalerKind::MetalFXSpatial => {
                #[cfg(target_os = "macos")]
                {
                    metalfx::MetalSpatialScaler::try_new(device, spatial)
                        .map(|s| Upscaler::Spatial(Box::new(s)))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (device, spatial);
                    None
                }
            }
            UpscalerKind::FSR2 => None,
        }
    }

    pub fn new(device: &wgpu::Device, kind: UpscalerKind, descriptor: UpscalerDescriptor) -> Self {
        Self::try_new(device, kind, descriptor)
            .unwrap_or_else(|| panic!("MetalFX scaler creation failed for {:?}", kind))
    }

    pub fn get_input_size(&self) -> (u32, u32) {
        match self {
            Upscaler::Spatial(s) => s.get_input_size(),
            Upscaler::Temporal(t) => t.get_input_size(),
        }
    }

    pub fn get_output_size(&self) -> (u32, u32) {
        match self {
            Upscaler::Spatial(s) => s.get_output_size(),
            Upscaler::Temporal(t) => t.get_output_size(),
        }
    }

    pub fn set_color_texture(&mut self, view: wgpu::TextureView) {
        match self {
            Upscaler::Spatial(s) => s.set_color_texture(view),
            Upscaler::Temporal(t) => t.set_color_texture(view),
        }
    }

    pub fn set_output_texture(&mut self, view: wgpu::TextureView) {
        match self {
            Upscaler::Spatial(s) => s.set_output_texture(view),
            Upscaler::Temporal(t) => t.set_output_texture(view),
        }
    }

    pub fn set_input_content_size(&mut self, width: u32, height: u32) {
        match self {
            Upscaler::Spatial(s) => s.set_input_content_size(width, height),
            Upscaler::Temporal(t) => t.set_input_content_size(width, height),
        }
    }

    pub fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Upscaler::Spatial(s) => s.dispatch(encoder),
            Upscaler::Temporal(t) => t.dispatch(encoder),
        }
    }

    pub fn as_temporal_mut(&mut self) -> Option<&mut dyn TemporalScaler> {
        match self {
            Upscaler::Temporal(t) => Some(t.as_mut()),
            _ => None,
        }
    }

    pub fn as_spatial_mut(&mut self) -> Option<&mut dyn SpatialScaler> {
        match self {
            Upscaler::Spatial(s) => Some(s.as_mut()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Jitter helpers (Halton sequence; matches Apple’s MetalFX temporal sample)
// ---------------------------------------------------------------------------

fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut result = 0.0;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}

/// Subpixel jitter in pixel space for temporal antialiasing / MetalFX temporal scaler,
/// matching [Apple’s sample](https://developer.apple.com/documentation/metalfx/applying-temporal-antialiasing-and-upscaling-using-metalfx)
/// (`halton(index, 2) - 0.5`, `halton(index, 3) - 0.5` with `index = frame % 32 + 1`).
pub fn temporal_antialiasing_jitter_pixels(frame_index: u32) -> (f32, f32) {
    let jitter_index = frame_index % 32 + 1;
    let x = halton(jitter_index, 2) - 0.5;
    let y = halton(jitter_index, 3) - 0.5;
    (x, y)
}

fn jitter_phase_count(render_width: i32, display_width: i32) -> i32 {
    let base_phase_count = 8.0_f32;
    let scale_factor = display_width as f32 / render_width as f32;
    let jitter_phase_count = base_phase_count * scale_factor.powi(2);
    jitter_phase_count as i32
}

/// Generic Halton jitter (different parameterization than [`temporal_antialiasing_jitter_pixels`]).
pub fn get_generic_jitter_offset(
    input_width: i32,
    output_width: i32,
    frame_index: u32,
) -> (f32, f32) {
    let phase_count = jitter_phase_count(input_width, output_width).max(2) as u32;
    let jitter_offset_x = halton(frame_index, phase_count);
    let jitter_offset_y = halton(frame_index, phase_count);
    (jitter_offset_x, jitter_offset_y)
}
