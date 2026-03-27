//! Simple forward-lit raster pass for the Cornell box from [`crate::geometry`].

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::geometry::{
    self, BlasVertex, MAT_EMISSIVE, MAT_GREEN, MAT_RED, MAT_WHITE, TriangleMeta,
};
use wgpu_scaler::temporal_antialiasing_jitter_pixels;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuUniforms {
    view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    render_resolution: [f32; 2],
    texel_jitter: [f32; 2],
    prev_texel_jitter: [f32; 2],
    motion_vector_frame_index: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    emissive: f32,
}

fn material_albedo(material_id: u32) -> [f32; 3] {
    match material_id {
        MAT_WHITE => [0.92, 0.92, 0.88],
        MAT_RED => [0.85, 0.2, 0.15],
        MAT_GREEN => [0.15, 0.75, 0.22],
        MAT_EMISSIVE => [1.0, 0.95, 0.75],
        _ => [0.5, 0.5, 0.5],
    }
}

fn expand_mesh(
    vertices: &[BlasVertex],
    indices: &[u16],
    meta: &[TriangleMeta],
) -> Vec<SceneVertex> {
    let mut out = Vec::with_capacity(indices.len());
    for (tri_i, tri) in indices.chunks_exact(3).enumerate() {
        let m = &meta[tri_i];
        let n = [m.normal[0], m.normal[1], m.normal[2]];
        let c = material_albedo(m.material_id);
        let emissive = if m.material_id == MAT_EMISSIVE {
            1.0
        } else {
            0.0
        };
        for &idx in tri {
            let p = vertices[idx as usize].pos;
            out.push(SceneVertex {
                position: p,
                normal: n,
                color: c,
                emissive,
            });
        }
    }
    out
}

/// Motion vectors follow Apple’s MetalFX sample: UV difference after un-jittering
/// (`ApplyingTemporalAntialiasingAndUpscalingUsingMetalFX`, `fragmentShaderTAA`).
const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    render_resolution: vec2<f32>,
    texel_jitter: vec2<f32>,
    prev_texel_jitter: vec2<f32>,
    motion_vector_frame_index: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) emissive: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) emissive: f32,
    @location(4) prev_clip: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = vec4<f32>(in.position, 1.0);
    out.clip_pos = u.view_proj * world;
    out.world_pos = in.position;
    out.normal = in.normal;
    out.color = in.color;
    out.emissive = in.emissive;
    out.prev_clip = u.prev_view_proj * world;
    return out;
}

fn shade_lit(normal: vec3<f32>, color: vec3<f32>, emissive: f32) -> vec3<f32> {
    let n = normalize(normal);
    let l = normalize(vec3<f32>(0.15, 0.92, 0.35));
    let ndl = max(dot(n, l), 0.0);
    let ambient = 0.12;
    var rgb = color * (ambient + 0.88 * ndl);
    if (emissive > 0.5) {
        rgb = rgb + color * 2.2;
    }
    return rgb;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade_lit(in.normal, in.color, in.emissive), 1.0);
}

/// Fragment stage: one `@builtin(position)` only (cannot reuse `VertexOutput` and add another).
struct FragmentMotionIn {
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) emissive: f32,
    @location(4) prev_clip: vec4<f32>,
}

struct MotionFragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
}

@fragment
fn fs_main_motion(in: FragmentMotionIn) -> MotionFragmentOut {
    var out: MotionFragmentOut;
    out.color = vec4<f32>(shade_lit(in.normal, in.color, in.emissive), 1.0);

    
    let scale = vec2<f32>(0.5, -0.5);
    let offset = 0.5;
    let uv = in.frag_coord.xy / u.render_resolution;
    let prev_uv = in.prev_clip.xy / in.prev_clip.w * scale + vec2<f32>(offset, offset);
    let uv_uj = uv - u.texel_jitter;
    let prev_uv_uj = prev_uv - u.prev_texel_jitter;
    let motion_vector = prev_uv_uj - uv_uj;
    
    out.motion = motion_vector;
    return out;
}
"#;

pub struct SceneRenderer {
    pipeline: wgpu::RenderPipeline,
    pipeline_motion: Option<wgpu::RenderPipeline>,
    with_motion_vectors: bool,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    width: u32,
    height: u32,
    color_format: wgpu::TextureFormat,
    prev_view_proj: Mat4,
    prev_texel_jitter: [f32; 2],
    motion_history_valid: bool,
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        with_motion_vectors: bool,
    ) -> Self {
        let (v, idx, meta) = geometry::build_cornell_mesh();
        let expanded = expand_mesh(&v, &idx, &meta);
        let vertex_count = expanded.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cornell vertices"),
            contents: bytemuck::cast_slice(&expanded),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene uniforms"),
            size: std::mem::size_of::<GpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let (depth_texture, depth_view) = create_depth(device, width, height);

        let vertex_buffers = &[wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Float32x3,
                3 => Float32,
            ],
        }];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let pipeline_motion = with_motion_vectors.then(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("scene pipeline motion"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: vertex_buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main_motion"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rg16Float,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                multiview_mask: None,
                cache: None,
            })
        });

        Self {
            pipeline,
            pipeline_motion,
            with_motion_vectors,
            uniform_buffer,
            bind_group,
            vertex_buffer,
            vertex_count,
            depth_texture,
            depth_view,
            width,
            height,
            color_format,
            prev_view_proj: Mat4::IDENTITY,
            prev_texel_jitter: [0.0, 0.0],
            motion_history_valid: false,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.motion_history_valid = false;
        let (tex, view) = create_depth(device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
    }

    /// Orbit camera; call once per frame before encoding the pass.
    ///
    /// When `jitter_for_temporal` is `true`, applies sub-pixel projection shear matching
    /// [Apple’s MetalFX temporal sample](https://developer.apple.com/documentation/metalfx/applying-temporal-antialiasing-and-upscaling-using-metalfx):
    /// Halton jitter in pixels, then `ndc = -2 * pixel_jitter / resolution` added to the third
    /// column of the projection matrix (same convention as `AAPLRenderer.updateJitterOffsets`).
    pub fn update_camera(&mut self, queue: &wgpu::Queue, frame: u32, jitter_for_temporal: bool) {
        let t = frame as f32 * 0.008;
        let eye = Vec3::new(
            0.5 + 0.85 * t.sin(),
            0.52 + 0.12 * (t * 1.3).cos(),
            -0.35 + 0.55 * t.cos(),
        );
        let target = Vec3::new(0.5, 0.45, 0.55);
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        let proj = Mat4::perspective_rh(
            55.0_f32.to_radians(),
            self.width.max(1) as f32 / self.height.max(1) as f32,
            0.05,
            10.0,
        );
        let rw = self.width.max(1) as f32;
        let rh = self.height.max(1) as f32;
        let (jx, jy) = if jitter_for_temporal {
            temporal_antialiasing_jitter_pixels(frame)
        } else {
            (0.0, 0.0)
        };
        // Match Apple: flip Y when converting pixel jitter to texel jitter for UV space.
        let texel_jitter = [jx / rw, -jy / rh];

        let proj = if jitter_for_temporal {
            let ndc_jx = -2.0 * jx / rw;
            let ndc_jy = -2.0 * jy / rh;
            let z = proj.z_axis + Vec4::new(ndc_jx, ndc_jy, 0.0, 0.0);
            Mat4::from_cols(proj.x_axis, proj.y_axis, z, proj.w_axis)
        } else {
            proj
        };
        let view_proj = proj * view;

        let motion_vector_frame_index = u32::from(self.motion_history_valid);

        let u = GpuUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            render_resolution: [rw, rh],
            texel_jitter,
            prev_texel_jitter: self.prev_texel_jitter,
            motion_vector_frame_index,
            _pad: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));

        self.prev_view_proj = view_proj;
        self.prev_texel_jitter = texel_jitter;
        self.motion_history_valid = true;
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_target: &wgpu::TextureView,
        motion_target: Option<&wgpu::TextureView>,
    ) {
        match (motion_target, self.pipeline_motion.as_ref()) {
            (Some(mv), Some(pm)) => {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene motion"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: color_target,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: mv,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pm);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.draw(0..self.vertex_count, 0..1);
            }
            (None, _) => {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.draw(0..self.vertex_count, 0..1);
            }
            (Some(_), None) => {
                panic!(
                    "motion render target provided but SceneRenderer was built without motion pipeline"
                );
            }
        }
    }

    /// Depth written during [`Self::render`], for feeding temporal upscalers.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Recreate pipeline when the color attachment format changes (e.g. swapchain format).
    pub fn set_color_format(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        with_motion_vectors: bool,
    ) {
        let format_changed = color_format != self.color_format;
        let motion_changed = with_motion_vectors != self.with_motion_vectors;
        if !format_changed && !motion_changed {
            return;
        }
        self.color_format = color_format;
        self.with_motion_vectors = with_motion_vectors;
        self.motion_history_valid = false;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        });
        self.bind_group = bind_group;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let vertex_buffers = &[wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Float32x3,
                3 => Float32,
            ],
        }];

        self.pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        self.pipeline_motion = with_motion_vectors.then(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("scene pipeline motion"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: vertex_buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main_motion"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rg16Float,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
    }
}

fn create_depth(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
