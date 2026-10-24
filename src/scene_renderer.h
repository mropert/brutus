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
	static constexpr uint32_t SHADOW_CASCADE_MAX = 4;
	// Task workgroups per bucket and pass, checked against the worst case of all draws at their largest LOD
	static constexpr uint32_t MAX_TASK_COMMANDS = 256 * 1024;

	struct DrawStatistics
	{
		std::array<uint32_t, MESH_LOD_COUNT> meshes;
		std::array<uint32_t, MESH_LOD_COUNT> indices;
	};

	SceneRenderer( renderer::Device& device,
				   renderer::Texture::Format output_format,
				   BindlessManager& bindless_manager,
				   renderer::PipelineManager& pipeline_manager );

	void add_mesh( const mesh& mesh, glm::vec3 translation, glm::quat rotation, glm::vec3 scale, const material& material );
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
				 uint32_t shadow_cascade_count,
				 bool show_shadow_cascades,
				 bool use_pbr,
				 bool use_occlusion_culling );

	const auto& get_statistics() const { return _statistics; }

private:
	static constexpr auto SAMPLE_COUNT = 4;
	static constexpr auto COLOR_FORMAT = renderer::Texture::Format::R16G16B16A16_SFLOAT;
	static constexpr auto DEPTH_FORMAT = renderer::Texture::Format::D32_SFLOAT;
	static constexpr auto SHADOW_DEPTH_FORMAT = renderer::Texture::Format::D16_UNORM;
	// The 100x100 benchmark crowd (~40k draws) needs ~12M bits
	static constexpr uint32_t MAX_MESHLET_VISIBILITY_BITS = 32 * 1024 * 1024;
	// Rasterizer bias of shadow casters, in D16 units (1/65536 of the cascade depth range)
	static constexpr float SHADOW_DEPTH_BIAS_CONSTANT = 1.f;
	static constexpr float SHADOW_DEPTH_BIAS_SLOPE = 2.f;

	// A bucket's part of a draw commands buffer, laid out for one render path
	struct DrawCommandsSegment
	{
		size_t commands_offset = 0;
		uint32_t capacity = 0; // Worst case number of commands for the current scene
	};
	using DrawCommandsSegments = std::array<DrawCommandsSegment, DRAW_BUCKETS_COUNT>;
	template <typename Commands>
	static DrawCommandsSegments make_empty_draw_commands_segments();
	// Grows a bucket's capacity, the following buckets' commands move accordingly
	template <typename Commands>
	static void add_draw_commands( DrawCommandsSegments& segments, size_t bucket, uint32_t count );
	const DrawCommandsSegments& get_draw_commands_segments( bool task_commands ) const
	{
		return task_commands ? _task_segments : _draw_mesh_segments;
	}
	// One indirect draw for all the commands of a bucket
	void draw_bucket( renderer::CommandBuffer& cmd, const renderer::Buffer& draw_commands, size_t bucket, bool use_mesh_shader ) const;

	void compute_draws( const BindlessManager& bindless_manager,
						const renderer::PipelineManager& pipelines,
						renderer::CommandBuffer& cmd,
						renderer::BufferHandle constants_address,
						renderer::BufferHandle draw_buffer_address,
						const renderer::Buffer& visibility_buffer,
						renderer::BufferHandle stats_buffer_address,
						const renderer::Buffer& draw_commands,
						float lod_target,
						bool early,
						bool meshlet_occlusion,
						bool task_commands ) const;
	// Matches MESHLET_PASS_* in mesh.task
	enum class MeshletPass : uint32_t
	{
		NONE = 0,
		EARLY,
		LATE
	};
	void draw_opaques( const BindlessManager& bindless_manager,
					   const renderer::PipelineManager& pipelines,
					   renderer::CommandBuffer& cmd,
					   renderer::BufferHandle constants_address,
					   renderer::BufferHandle draw_buffer_address,
					   bool use_mesh_shader,
					   MeshletPass meshlet_pass ) const;
	void draw_opaque_meshlets( const BindlessManager& bindless_manager,
							   const renderer::PipelineManager& pipelines,
							   renderer::CommandBuffer& cmd,
							   renderer::BufferHandle constants_address,
							   renderer::BufferHandle draw_buffer_address,
							   MeshletPass meshlet_pass ) const;
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

	struct ShadowCascades
	{
		std::array<glm::mat4, SHADOW_CASCADE_MAX> view_projections;
		glm::vec4 splits; // Far view depth of each cascade
		glm::vec4 texel_sizes; // World size of a shadow map texel in each cascade
		uint32_t count = 0;
	};

	ShadowCascades draw_shadowmap( const BindlessManager& bindless_manager,
								   const renderer::PipelineManager& pipelines,
								   renderer::CommandBuffer& cmd,
								   renderer::BufferHandle draw_buffer_address,
								   renderer::BufferHandle stats_buffer_address,
								   const Camera& camera,
								   glm::vec3 main_light,
								   float shadow_distance,
								   uint32_t cascade_count,
								   bool use_mesh_shader );

	struct MeshDraw
	{
		glm::vec3 translation;
		mesh_handle mesh;
		glm::quat rotation;
		glm::vec3 scale;
		material_handle material;
		uint32_t meshlet_visibility_offset;
		uint32_t padding[ 3 ]; // std430 rounds the GPU struct size to its 16 bytes alignment
	};
	static_assert( sizeof( MeshDraw ) == 64 );

	renderer::Extent2D _extent;
	renderer::Extent2D _depth_pyramid_extent;
	renderer::raii::Texture _color_texture;
	renderer::raii::TextureView _color_texture_view;
	renderer::raii::Texture _depth_texture;
	renderer::raii::TextureView _depth_texture_view;
	renderer::BindlessTexture _resolved_color_texture;
	renderer::BindlessTexture _resolved_depth_texture;
	renderer::BindlessTexture _depth_pyramid;
	renderer::BindlessTexture _shadowmap; // One layer per cascade
	std::array<renderer::raii::TextureView, SHADOW_CASCADE_MAX> _shadowmap_layer_views;
	renderer::raii::Buffer _constant_buffers;
	renderer::raii::Buffer _shadowmap_constant_buffers;
	renderer::raii::Buffer _draw_buffers;
	renderer::raii::Buffer _draw_visibility_buffer;
	renderer::raii::Buffer _shadowmap_visibility_buffer;
	// One bit per main LOD meshlet of each draw, not reset when the scene changes: the late pass corrects stale bits
	renderer::raii::Buffer _meshlet_visibility_buffer;
	uint32_t _meshlet_visibility_bits = 0;
	// Capacities are the worst case for the current scene: one command per draw for the vertex path, its largest LOD's
	// task workgroups for the mesh shader path
	DrawCommandsSegments _draw_mesh_segments;
	DrawCommandsSegments _task_segments;
	renderer::raii::Buffer _draw_commands_buffer;
	renderer::raii::Buffer _shadowmap_draw_commands_buffer;
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
	renderer::PipelineHandle _compute_tasks_early_pipeline;
	renderer::PipelineHandle _compute_tasks_late_pipeline;
	renderer::PipelineHandle _depth_pyramid_pipeline;
	renderer::PipelineHandle _grid_pipeline;
	renderer::PipelineHandle _tonemap_pipeline;
	std::vector<MeshDraw> _scene;
	DrawStatistics _statistics;
	int _frames = 0;
	bool _needs_rebuild = false;
	bool _draw_indirect_count;
};
