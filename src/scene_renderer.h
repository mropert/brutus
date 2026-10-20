#pragma once

#include "asset.h"

#include <array>
#include <renderer/pipeline_manager.h>
#include <renderer/texture.h>

struct Camera;

class SceneRenderer
{
public:
	static constexpr auto DRAW_BUCKETS_COUNT = std::to_underlying( material_type::COUNT );
	static constexpr float FOV = 70.f;

	struct DrawStatistics
	{
		std::array<uint32_t, MESH_LOD_COUNT> meshes;
		std::array<uint32_t, MESH_LOD_COUNT> indices;
	};

	SceneRenderer( renderer::Device& device,
				   renderer::Texture::Format output_format,
				   BindlessManager& bindless_manager,
				   renderer::PipelineManager& pipeline_manager );

	void add_mesh( mesh_handle mesh, glm::vec3 translation, glm::quat rotation, glm::vec3 scale, material_handle material );
	void clear_meshes();

	void render( const BindlessManager& bindless_manager,
				 const renderer::PipelineManager& pipelines,
				 renderer::CommandBuffer& cmd,
				 renderer::Texture output_texture,
				 renderer::TextureView output_texture_view,
				 const renderer::Buffer* index_buffer, // null -> use mesh shaders
				 const Camera& camera,
				 glm::vec4 main_light,
				 float sky_illumination,
				 float lod_pixel_tolerance,
				 float shadow_distance,
				 bool use_pbr,
				 bool use_occlusion_culling );

	const auto& get_statistics() const { return _statistics; }

private:
	static constexpr auto SAMPLE_COUNT = 4;
	static constexpr auto COLOR_FORMAT = renderer::Texture::Format::R16G16B16A16_SFLOAT;
	static constexpr auto DEPTH_FORMAT = renderer::Texture::Format::D32_SFLOAT;

	void compute_draws( const BindlessManager& bindless_manager,
						const renderer::PipelineManager& pipelines,
						renderer::CommandBuffer& cmd,
						renderer::BufferHandle constants_address,
						renderer::BufferHandle draw_buffer_address,
						const renderer::Buffer& visibility_buffer,
						renderer::BufferHandle stats_buffer_address,
						const std::array<renderer::raii::Buffer, DRAW_BUCKETS_COUNT>& draw_commands,
						float lod_target,
						bool early ) const;
	void draw_opaques( const BindlessManager& bindless_manager,
					   const renderer::PipelineManager& pipelines,
					   renderer::CommandBuffer& cmd,
					   renderer::BufferHandle constants_address,
					   renderer::BufferHandle draw_buffer_address,
					   bool use_mesh_shader ) const;
	void draw_opaque_meshlets( const BindlessManager& bindless_manager,
							   const renderer::PipelineManager& pipelines,
							   renderer::CommandBuffer& cmd,
							   renderer::BufferHandle constants_address,
							   renderer::BufferHandle draw_buffer_address ) const;
	void draw_opaque_meshes( const BindlessManager& bindless_manager,
							 const renderer::PipelineManager& pipelines,
							 renderer::CommandBuffer& cmd,
							 renderer::BufferHandle constants_address,
							 renderer::BufferHandle draw_buffer_address ) const;

	void draw_transparent_meshlets( const BindlessManager& bindless_manager,
									const renderer::PipelineManager& pipelines,
									renderer::CommandBuffer& cmd,
									renderer::BufferHandle constants_address,
									renderer::BufferHandle draw_buffer_address ) const;
	void draw_transparent_meshes( const BindlessManager& bindless_manager,
								  const renderer::PipelineManager& pipelines,
								  renderer::CommandBuffer& cmd,
								  renderer::BufferHandle constants_address,
								  renderer::BufferHandle draw_buffer_address ) const;

	void draw_shadowmap( const BindlessManager& bindless_manager,
						 const renderer::PipelineManager& pipelines,
						 renderer::CommandBuffer& cmd,
						 renderer::BufferHandle draw_buffer_address,
						 renderer::BufferHandle stats_buffer_address,
						 const Camera& camera,
						 glm::vec3 main_light,
						 float shadow_distance );

	struct MeshDraw
	{
		glm::vec3 translation;
		mesh_handle mesh;
		glm::quat rotation;
		glm::vec3 scale;
		material_handle material;
	};

	renderer::Extent2D _extent;
	renderer::Extent2D _depth_pyramid_extent;
	renderer::raii::Texture _color_texture;
	renderer::raii::TextureView _color_texture_view;
	renderer::raii::Texture _depth_texture;
	renderer::raii::TextureView _depth_texture_view;
	renderer::BindlessTexture _resolved_color_texture;
	renderer::BindlessTexture _resolved_depth_texture;
	renderer::BindlessTexture _depth_pyramid;
	renderer::BindlessTexture _shadowmap;
	renderer::raii::Buffer _constant_buffers;
	renderer::raii::Buffer _shadowmap_constant_buffers;
	renderer::raii::Buffer _draw_buffers;
	renderer::raii::Buffer _draw_visibility_buffer;
	renderer::raii::Buffer _shadowmap_visibility_buffer;
	std::array<renderer::raii::Buffer, DRAW_BUCKETS_COUNT> _draw_commands_buffers;
	std::array<renderer::raii::Buffer, DRAW_BUCKETS_COUNT> _shadowmap_draw_commands_buffers;
	renderer::raii::Buffer _draw_statistics_buffer; // GPU only, accumulated by culling
	std::array<renderer::raii::Buffer, renderer::MAX_FRAMES_IN_FLIGHT> _draw_statistics_readback_buffers;
	renderer::PipelineHandle _mesh_pipeline;
	renderer::PipelineHandle _mesh_alpha_clip_pipeline;
	renderer::PipelineHandle _mesh_alpha_blend_pipeline;
	renderer::PipelineHandle _shadowmap_mesh_pipeline;
	renderer::PipelineHandle _meshlet_pipeline;
	renderer::PipelineHandle _meshlet_alpha_clip_pipeline;
	renderer::PipelineHandle _meshlet_alpha_blend_pipeline;
	renderer::PipelineHandle _shadowmap_meshlet_pipeline;
	renderer::PipelineHandle _compute_draws_early_pipeline;
	renderer::PipelineHandle _compute_draws_late_pipeline;
	renderer::PipelineHandle _depth_pyramid_pipeline;
	renderer::PipelineHandle _grid_pipeline;
	renderer::PipelineHandle _tonemap_pipeline;
	std::vector<MeshDraw> _scene;
	DrawStatistics _statistics;
	int _frames = 0;
	bool _needs_rebuild = false;
	bool _draw_indirect_count;
	glm::mat4 _shadow_mvp;
};
