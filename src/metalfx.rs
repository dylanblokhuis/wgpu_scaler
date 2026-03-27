//! MetalFX spatial and temporal scalers (Metal backend only).

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLPixelFormat;
use objc2_metal_fx::{
    MTLFXSpatialScaler, MTLFXSpatialScalerBase, MTLFXSpatialScalerColorProcessingMode,
    MTLFXSpatialScalerDescriptor, MTLFXTemporalScaler, MTLFXTemporalScalerBase,
    MTLFXTemporalScalerDescriptor,
};
use wgpu::{AstcBlock, AstcChannel, wgc::api::Metal};

use crate::{SpatialScaler, SpatialUpscalerDescriptor, TemporalScaler, TemporalUpscalerDescriptor};

pub struct MetalTemporalScaler {
    inner: Retained<ProtocolObject<dyn MTLFXTemporalScaler>>,
}

impl MetalTemporalScaler {
    pub(crate) fn try_new(
        device: &wgpu::Device,
        descriptor: TemporalUpscalerDescriptor,
    ) -> Option<Self> {
        let metal_device = unsafe { device.as_hal::<Metal>() }?;

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
            desc.setAutoExposureEnabled(descriptor.auto_exposure);
            desc.setRequiresSynchronousInitialization(descriptor.requires_synchronous_initialization);
            desc.setReactiveMaskTextureEnabled(descriptor.reactive_mask_enabled);
            if descriptor.reactive_mask_enabled {
                desc.setReactiveMaskTextureFormat(map_format(
                    descriptor.reactive_mask_texture_format,
                ));
            }
            desc.setInputContentPropertiesEnabled(descriptor.input_content_properties_enabled);
            if descriptor.input_content_properties_enabled {
                desc.setInputContentMinScale(descriptor.input_content_min_scale);
                desc.setInputContentMaxScale(descriptor.input_content_max_scale);
            }

            desc.newTemporalScalerWithDevice(metal_device.raw_device())?
        };

        unsafe {
            temporal_scaler.setInputContentWidth(descriptor.input_width as _);
            temporal_scaler.setInputContentHeight(descriptor.input_height as _);
        }

        Some(MetalTemporalScaler {
            inner: temporal_scaler,
        })
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
                color_texture
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
                motion_texture
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
                depth_texture
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
                output_texture
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

    fn set_input_content_size(&mut self, width: u32, height: u32) {
        unsafe {
            self.inner.setInputContentWidth(width as _);
            self.inner.setInputContentHeight(height as _);
        }
    }

    fn set_exposure_texture(&mut self, exposure_texture: Option<wgpu::TextureView>) {
        unsafe {
            match exposure_texture.as_ref() {
                Some(v) => self.inner.setExposureTexture(Some(
                    v.texture()
                        .as_hal::<Metal>()
                        .unwrap()
                        .raw_handle(),
                )),
                None => self.inner.setExposureTexture(None),
            }
        };
    }

    fn set_reactive_mask_texture(&mut self, mask: Option<wgpu::TextureView>) {
        unsafe {
            match mask.as_ref() {
                Some(v) => self.inner.setReactiveMaskTexture(Some(
                    v.texture()
                        .as_hal::<Metal>()
                        .unwrap()
                        .raw_handle(),
                )),
                None => self.inner.setReactiveMaskTexture(None),
            }
        };
    }

    fn set_pre_exposure(&mut self, pre_exposure: f32) {
        unsafe { self.inner.setPreExposure(pre_exposure) };
    }

    fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder) {
        unsafe {
            encoder.as_hal_mut::<Metal, _, _>(|inner| {
                let handle = inner.unwrap().raw_command_buffer().unwrap();
                self.inner.encodeToCommandBuffer(handle);
            });
        };
    }
}

pub struct MetalSpatialScaler {
    inner: Retained<ProtocolObject<dyn MTLFXSpatialScaler>>,
}

impl MetalSpatialScaler {
    pub(crate) fn try_new(
        device: &wgpu::Device,
        descriptor: SpatialUpscalerDescriptor,
    ) -> Option<Self> {
        let metal_device = unsafe { device.as_hal::<Metal>() }?;

        let spatial_scaler = unsafe {
            let desc = MTLFXSpatialScalerDescriptor::new();
            desc.setInputWidth(descriptor.input_width as _);
            desc.setInputHeight(descriptor.input_height as _);
            desc.setOutputWidth(descriptor.output_width as _);
            desc.setOutputHeight(descriptor.output_height as _);
            desc.setColorTextureFormat(map_format(descriptor.color_texture_format));
            desc.setOutputTextureFormat(map_format(descriptor.output_texture_format));

            desc.setColorProcessingMode(color_processing_mode(descriptor.color_texture_format));

            desc.newSpatialScalerWithDevice(metal_device.raw_device())?
        };

        unsafe {
            spatial_scaler.setInputContentWidth(descriptor.input_width as _);
            spatial_scaler.setInputContentHeight(descriptor.input_height as _);
        }

        Some(MetalSpatialScaler {
            inner: spatial_scaler,
        })
    }
}

fn color_processing_mode(format: wgpu::TextureFormat) -> MTLFXSpatialScalerColorProcessingMode {
    if format.is_srgb() {
        MTLFXSpatialScalerColorProcessingMode::Perceptual
    } else if is_hdr(format) {
        MTLFXSpatialScalerColorProcessingMode::HDR
    } else {
        MTLFXSpatialScalerColorProcessingMode::Linear
    }
}

impl SpatialScaler for MetalSpatialScaler {
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

    fn set_color_texture(&mut self, color_texture: wgpu::TextureView) {
        unsafe {
            self.inner.setColorTexture(Some(
                color_texture
                    .texture()
                    .as_hal::<Metal>()
                    .unwrap()
                    .raw_handle(),
            ))
        }
    }

    fn set_output_texture(&mut self, output_texture: wgpu::TextureView) {
        unsafe {
            self.inner.setOutputTexture(Some(
                output_texture
                    .texture()
                    .as_hal::<Metal>()
                    .unwrap()
                    .raw_handle(),
            ))
        }
    }

    fn set_input_content_size(&mut self, width: u32, height: u32) {
        unsafe {
            self.inner.setInputContentWidth(width as _);
            self.inner.setInputContentHeight(height as _);
        }
    }

    fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder) {
        unsafe {
            encoder.as_hal_mut::<Metal, _, _>(|inner| {
                let handle = inner.unwrap().raw_command_buffer().unwrap();
                self.inner.encodeToCommandBuffer(handle);
            });
        };
    }
}

fn is_hdr(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
    )
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
