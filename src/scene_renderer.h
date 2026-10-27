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
	// Culling sorts draws into one list per bucket and LOD so that passes draw LOD 0 first, roughly front to back. Lists
	// are packed in their bucket's commands once their sizes are known (see mesh_commands.comp)
	static constexpr size_t DRAW_LISTS_COUNT = DRAW_BUCKETS_COUNT * MESH_LOD_COUNT;
	// Vertical, in radians: 50.70 degrees. It used to be 70 degrees passed as radians by accident, which gave this angle;
	// kept so that reference images and benchmarks stay comparable
	static constexpr float FOV = 0.884962f;
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
				 renderer::TimestampQuery timings_pool,
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
	// Must match SHADOW_CASCADE_BLEND_RANGE in shadow.glsl
	static constexpr float SHADOW_CASCADE_BLEND_RANGE = 0.25f;
	// How far from a receiver's position, in texels of its cascade, a shadow lookup reads: 1.5 texels of normal offset
	// and the 5x5 tent read with bilinear taps (3.5 texels), plus one of slack
	static constexpr float SHADOW_CASTER_MARGIN_TEXELS = 6.f;

	// A bucket's part of a draw commands buffer, laid out for one render path. Its LOD lists are packed inside
	struct DrawCommandsSegment
	{
		size_t commands_offset = 0;
		uint32_t capacity = 0; // Worst case number of commands for the current scene
	};
	using DrawCommandsSegments = std::array<DrawCommandsSegment, DRAW_BUCKETS_COUNT>;
	static constexpr size_t calc_draw_list( size_t bucket, size_t lod ) { return bucket * MESH_LOD_COUNT + lod; }
	template <typename Commands>
	static DrawCommandsSegments make_empty_draw_commands_segments();
	// Grows a bucket's capacity, the following buckets' commands move accordingly
	template <typename Commands>
	static void add_draw_commands( DrawCommandsSegments& segments, size_t bucket, uint32_t count );
	const DrawCommandsSegments& get_draw_commands_segments( bool task_commands ) const
	{
		return task_commands ? _task_segments : _draw_mesh_segments;
	}
	// Mesh shader path: one indirect draw for the commands of a list, the task shader reads where they start
	void draw_list( renderer::CommandBuffer& cmd, const renderer::Buffer& draw_commands, size_t list ) const;
	// Vertex path: one indirect multi draw for all the lists of a bucket, in LOD order
	void draw_bucket( renderer::CommandBuffer& cmd, const renderer::Buffer& draw_commands, size_t bucket ) const;

	// Culls the draws of a pass into draw_commands, with the barriers around it: culling counts each list and gives each
	// draw a slot, then the commands are written in their packed lists
	void compute_draws( const BindlessManager& bindless_manager,
						const renderer::PipelineManager& pipelines,
						renderer::CommandBuffer& cmd,
						renderer::BufferHandle pass_constants_address,
						renderer::BufferHandle draw_buffer_address,
						const renderer::Buffer& visibility_buffer,
						renderer::BufferHandle stats_buffer_address,
						const renderer::Buffer& draw_commands,
						float lod_target,
						bool early,
						bool meshlet_occlusion,
						bool task_commands ) const;
	// The steps of compute_draws without barriers, for callers that batch several passes
	void clear_draw_commands( renderer::CommandBuffer& cmd, const renderer::Buffer& draw_commands, bool task_commands ) const;
	void dispatch_culling( const BindlessManager& bindless_manager,
						   const renderer::PipelineManager& pipelines,
						   renderer::CommandBuffer& cmd,
						   renderer::BufferHandle pass_constants_address,
						   renderer::BufferHandle draw_buffer_address,
						   const renderer::Buffer& visibility_buffer,
						   renderer::BufferHandle stats_buffer_address,
						   const renderer::Buffer& draw_commands,
						   float lod_target,
						   bool early,
						   bool meshlet_occlusion,
						   bool task_commands ) const;
	void dispatch_draw_commands( const BindlessManager& bindless_manager,
								 const renderer::PipelineManager& pipelines,
								 renderer::CommandBuffer& cmd,
								 renderer::BufferHandle pass_constants_address,
								 renderer::BufferHandle draw_buffer_address,
								 const renderer::Buffer& draw_commands,
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
					   renderer::BufferHandle frame_constants_address,
					   renderer::BufferHandle pass_constants_address,
					   renderer::BufferHandle draw_buffer_address,
					   bool use_mesh_shader,
					   MeshletPass meshlet_pass ) const;
	void draw_opaque_meshlets( const BindlessManager& bindless_manager,
							   const renderer::PipelineManager& pipelines,
							   renderer::CommandBuffer& cmd,
							   renderer::BufferHandle frame_constants_address,
							   renderer::BufferHandle pass_constants_address,
							   renderer::BufferHandle draw_buffer_address,
							   MeshletPass meshlet_pass ) const;
	void draw_opaque_meshes( const BindlessManager& bindless_manager,
							 const renderer::PipelineManager& pipelines,
							 renderer::CommandBuffer& cmd,
							 renderer::BufferHandle frame_constants_address,
							 renderer::BufferHandle pass_constants_address,
							 renderer::BufferHandle draw_buffer_address ) const;

	void draw_transparent_meshlets( const BindlessManager& bindless_manager,
									const renderer::PipelineManager& pipelines,
									renderer::CommandBuffer& cmd,
									renderer::BufferHandle frame_constants_address,
									renderer::BufferHandle pass_constants_address,
									renderer::BufferHandle draw_buffer_address ) const;
	void draw_transparent_meshes( const BindlessManager& bindless_manager,
								  const renderer::PipelineManager& pipelines,
								  renderer::CommandBuffer& cmd,
								  renderer::BufferHandle frame_constants_address,
								  renderer::BufferHandle pass_constants_address,
								  renderer::BufferHandle draw_buffer_address ) const;

	struct ShadowCascades
	{
		std::array<glm::mat4, SHADOW_CASCADE_MAX> view_projections;
		glm::vec4 splits; // Far view depth of each cascade
		glm::vec4 texel_sizes; // World size of a shadow map texel in each cascade
		uint32_t count = 0;
	};

	// Fits the cascades to the camera and writes their pass constants for this frame
	ShadowCascades setup_shadow_cascades( const Camera& camera, glm::vec3 main_light, float shadow_distance, uint32_t cascade_count );
	void draw_shadowmap( const BindlessManager& bindless_manager,
						 const renderer::PipelineManager& pipelines,
						 renderer::CommandBuffer& cmd,
						 renderer::TimestampQuery timings_pool,
						 renderer::BufferHandle frame_constants_address,
						 renderer::BufferHandle draw_buffer_address,
						 renderer::BufferHandle stats_buffer_address,
						 const ShadowCascades& cascades,
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
	renderer::raii::Buffer _frame_constant_buffers;
	renderer::raii::Buffer _pass_constant_buffers;
	renderer::raii::Buffer _shadowmap_pass_constant_buffers; // One per cascade
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
	std::array<renderer::raii::Buffer, SHADOW_CASCADE_MAX> _shadowmap_draw_commands_buffers; // One per cascade
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
	renderer::PipelineHandle _draw_commands_pipeline;
	renderer::PipelineHandle _task_commands_pipeline;
	renderer::PipelineHandle _depth_pyramid_pipeline;
	renderer::PipelineHandle _grid_pipeline;
	renderer::PipelineHandle _tonemap_pipeline;
	std::vector<MeshDraw> _scene;
	DrawStatistics _statistics;
	int _frames = 0;
	bool _needs_rebuild = false;
	bool _draw_indirect_count;
};
