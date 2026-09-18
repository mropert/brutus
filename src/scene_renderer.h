#pragma once

#include "asset.h"

#include <array>
#include <renderer/pipeline_manager.h>
#include <renderer/texture.h>

struct Camera;

class SceneRenderer
{
public:
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
				 bool use_pbr,
				 bool use_occlusion_culling );

private:
	static constexpr auto SAMPLE_COUNT = 4;
	static constexpr auto COLOR_FORMAT = renderer::Texture::Format::R16G16B16A16_SFLOAT;
	static constexpr auto DEPTH_FORMAT = renderer::Texture::Format::D32_SFLOAT;

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
	renderer::raii::Buffer _constant_buffers;
	renderer::raii::Buffer _draw_buffers;
	renderer::raii::Buffer _draw_visibility_buffer;
	renderer::raii::Buffer _draw_commands_buffer;
	renderer::PipelineHandle _mesh_pipeline;
	renderer::PipelineHandle _meshlet_pipeline;
	renderer::PipelineHandle _compute_draws_early_pipeline;
	renderer::PipelineHandle _compute_draws_late_pipeline;
	renderer::PipelineHandle _depth_pyramid_pipeline;
	renderer::PipelineHandle _grid_pipeline;
	renderer::PipelineHandle _tonemap_pipeline;
	std::vector<MeshDraw> _scene;
	int _frames = 0;
	bool _needs_rebuild = false;
	bool _draw_indirect_count;
};
