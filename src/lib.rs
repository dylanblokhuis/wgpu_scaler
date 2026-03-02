//! wgpu_scaler: cross-backend upscaling (MetalFX temporal, and more).

use std::ops::{Deref, DerefMut};

#[cfg(target_os = "macos")]
mod metalfx;

/// Identifies an upscaler implementation (MetalFX temporal, FSR, NIS, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpscalerKind {
    /// MetalFX Temporal Scaler (Metal backend only).
    MetalFxTemporal,
    MetalFxSpatial,
    FSR2,
}

/// Descriptor used to create any supported upscaler.
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

/// Returns which upscalers are supported for the given adapter (e.g. by backend).
pub fn supported_upscalers(adapter: &wgpu::Adapter) -> Vec<UpscalerKind> {
    let info = adapter.get_info();
    let mut out = Vec::new();
    let os = std::env::consts::OS;
    match info.backend {
        wgpu::Backend::Metal => {
            out.push(UpscalerKind::MetalFxTemporal);
        }
        wgpu::Backend::Vulkan => {

            // technically we can support:
            // - FSR
            //
        }
        _ => {}
    }
    out
}

/// Wrapper that can hold any concrete upscaler and is created from a generic descriptor.
pub struct Upscaler(Box<dyn TemporalScaler>);

impl Deref for Upscaler {
    type Target = Box<dyn TemporalScaler>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Upscaler {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Upscaler {
    /// Create an upscaler of the given kind with the given descriptor.
    /// Returns `None` if the kind is not supported for the device (e.g. wrong backend).
    pub fn new(device: &wgpu::Device, kind: UpscalerKind, descriptor: UpscalerDescriptor) -> Self {
        match kind {
            UpscalerKind::MetalFxTemporal => Upscaler(Box::new(metalfx::MetalTemporalScaler::new(
                device, descriptor,
            ))),
            UpscalerKind::MetalFxSpatial => Upscaler(Box::new(metalfx::MetalSpatialScaler::new(
                device, descriptor,
            ))),
            _ => todo!(),
        }
    }
}

/// Trait implemented by all temporal upscaler backends.
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
    fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder);
}

// ---------------------------------------------------------------------------
// Jitter helpers (generic, backend-agnostic)
// ---------------------------------------------------------------------------

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

/// Generic jitter offset for temporal upscaling (Halton sequence).
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
