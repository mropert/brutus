#include "scene_renderer.h"

#include "camera.h"
#include "gpu_timing.h"

#include <algorithm>
#include <numeric>
#include <renderer/command_buffer.h>
#include <renderer/details/profiler.h>
#include <renderer/device.h>

struct SceneRenderer::draw_meshes_constants
{
	renderer::BufferHandle frame_constants;
	renderer::BufferHandle pass_constants;
	renderer::BufferHandle draws_buffer;
	renderer::BufferHandle commands_buffer;
	// Task shader only
	renderer::BufferHandle meshlet_visibility;
	renderer::BindlessSampledTexture depth_pyramid;
	uint32_t meshlet_pass = 0;
	uint32_t draw_bin = 0;
};

namespace
{
	uint32_t calc_group_count( uint32_t count )
	{
		return ( count + 32 - 1 ) / 32;
	}

	uint32_t round_pow2( uint32_t v )
	{
		uint32_t r = 1;

		while ( r * 2 < v )
			r *= 2;

		return r;
	}

	// Material draw buckets come in pairs: single sided first, then double sided
	renderer::Pipeline::CullMode bucket_cull_mode( int index_in_pair )
	{
		return index_in_pair == 0 ? renderer::Pipeline::CullMode::BACK : renderer::Pipeline::CullMode::NONE;
	}

	static constexpr auto MAX_DRAWS = 65536;

	// Vertex shader path, one per draw
	struct gpu_draw_mesh_command
	{
		uint32_t draw_id;
		uint32_t lod;
		VkDrawIndexedIndirectCommand draw_cmd;
	};
	struct gpu_draw_mesh_commands
	{
		std::array<uint32_t, SceneRenderer::DRAW_BINS_COUNT> bin_counts;
		// Draw count of all the bins of a bucket, for its indirect multi draw
		std::array<uint32_t, SceneRenderer::DRAW_BUCKETS_COUNT> bucket_counts;
		// Commands are stored as one array, each bucket starting at the sum of the previous buckets' capacities
		std::array<gpu_draw_mesh_command, MAX_DRAWS> commands;
	};

	// Mesh shader path, one per task workgroup, MeshTaskCommand in mesh.glsl
	struct gpu_task_command
	{
		uint32_t draw_id;
		uint32_t first_meshlet; // Top bit: drawn early
	};
	struct gpu_task_header
	{
		VkDrawMeshTasksIndirectCommandEXT dispatch;
		uint32_t draw_count; // Always 1 for now
		uint32_t first_command;
	};

	struct gpu_task_commands
	{
		std::array<gpu_task_header, SceneRenderer::DRAW_BINS_COUNT> headers;
		std::array<gpu_task_command, SceneRenderer::MAX_TASK_COMMANDS> commands;
	};
	union gpu_draw_commands
	{
		gpu_draw_mesh_commands meshes;
		gpu_task_commands tasks;
	};

	// Where culling put a draw, DrawSlot in draw_commands.glsl
	struct gpu_draw_slot
	{
		uint32_t bin;
		uint32_t offset_in_bin;
	};
	struct gpu_draw_commands_buffer
	{
		gpu_draw_commands commands;
		std::array<gpu_draw_slot, MAX_DRAWS> slots;
	};

	template <typename Commands>
	constexpr size_t calc_header_offset( size_t bin )
	{
		return offsetof( Commands, headers ) + bin * sizeof( typename decltype( Commands::headers )::value_type );
	}

	// Index of a bucket's first command in the commands array, from its segment offset in bytes
	template <typename Commands>
	uint32_t calc_first_command( size_t commands_offset )
	{
		using Command = typename decltype( Commands::commands )::value_type;
		return static_cast<uint32_t>( ( commands_offset - offsetof( Commands, commands ) ) / sizeof( Command ) );
	}

	struct grid_constants
	{
		glm::aligned_vec3 grid_color = { 0.1f, 0.1f, 0.1f };
		renderer::BufferHandle frame_constants;
		renderer::BufferHandle pass_constants;
		float near_distance = 50.f;
		float far_distance = 60.f;
	};

	struct depth_pyramid_constants
	{
		renderer::BindlessSampledTexture depth_in;
		renderer::BindlessStorageTexture depth_out;
		glm::aligned_vec2 size;
	};

	struct depth_range_constants
	{
		renderer::BufferHandle depth_range;
		renderer::BindlessSampledTexture depth;
	};

	// DepthRange in depth_range.glsl
	struct gpu_depth_range
	{
		uint32_t nearest;
		uint32_t farthest;
	};

	struct shadow_cascades_constants
	{
		renderer::BufferHandle frame_constants;
		renderer::BufferHandle camera_pass_constants;
		renderer::BufferHandle depth_range;
		renderer::BufferHandle shadow_cascades;
		renderer::BufferHandle cascades_pass_constants; // An array of SHADOW_CASCADE_MAX
		uint32_t shadowmap_resolution;
	};

	struct compute_draws_constants
	{
		renderer::BufferHandle draws_buffer;
		renderer::BufferHandle commands_buffer;
		renderer::BufferHandle slots_buffer;
		renderer::BufferHandle visibility_buffer;
		renderer::BufferHandle statistics_buffer;
		renderer::BufferHandle pass_constants;
		uint32_t draws_count;
		renderer::BindlessSampledTexture depth_pyramid;
		uint32_t meshlet_occlusion;
	};

	struct draw_commands_constants
	{
		renderer::BufferHandle draws_buffer;
		renderer::BufferHandle commands_buffer;
		renderer::BufferHandle slots_buffer;
		renderer::BufferHandle pass_constants;
		uint32_t draws_count;
		std::array<uint32_t, SceneRenderer::DRAW_BUCKETS_COUNT> bucket_first_commands;
	};

	glm::vec4 normalize_plane( const glm::vec4& plane )
	{
		return plane / glm::length( glm::vec3( plane ) );
	}

	// FrameConstants in constants.glsl. glm::aligned_vec3 takes 16 bytes where std430 lets a scalar follow a vec3, so
	// they're only followed by 16 bytes aligned members
	struct frame_constants
	{
		// Main camera
		glm::aligned_vec3 camera_position;
		glm::aligned_vec3 camera_forward;

		// Lighting
		glm::aligned_vec4 main_light; // xyz the direction the light travels, w its intensity
		float sky_to_sun = 0.f;
		uint32_t pbr = 0;

		// Main light shadows
		renderer::BindlessSampledTexture shadow_map;
		uint32_t shadow_cascade_count = 0;
		float shadow_distance = 0.f; // View depth up to which the main light casts shadows, the cascades may stop before
		renderer::BufferHandle shadow_cascades;

		// Debug
		uint32_t show_shadow_cascades = 0;
	};

	// ShadowCascades in constants.glsl, written by shadow_cascades.comp
	struct gpu_shadow_cascades
	{
		std::array<glm::mat4, SceneRenderer::SHADOW_CASCADE_MAX> view_projection_matrices;
		glm::vec4 splits;
		glm::vec4 texel_sizes;
	};
	static_assert( SceneRenderer::SHADOW_CASCADE_MAX == 4, "splits holds one split per vec4 component" );

	// PassConstants in constants.glsl, same layout rules
	struct pass_constants
	{
		// View
		glm::mat4 view_projection_matrix;
		glm::mat4 projection_matrix;
		glm::mat4 view_matrix;
		glm::mat4 inverse_view_matrix;
		glm::aligned_vec3 view_position;
		std::array<glm::aligned_vec4, 6> frustum;

		// Render target
		glm::aligned_vec2 viewport_size; // In pixels
		uint32_t sample_count = 1; // Must use the standard sample locations

		// Culling selects the coarsest LOD whose error is under lod_target, scaled by distance for perspective projections
		float lod_target = 0.f;

		// Shadow passes only, see shadow_caster_visible() in constants.glsl
		uint32_t shadow_pass = 0; // Non-zero when rendering a shadow map: use shadow (depth-only) LODs
		std::array<glm::aligned_vec4, 6> shadow_receiver_planes;
		glm::aligned_vec4 shadow_caster_sweep;
	};
	// shadow_cascades.comp writes the cascades' constants as a struct array, whose stride is rounded to 16 bytes
	static_assert( sizeof( pass_constants ) % 16 == 0 );

	pass_constants
	build_main_pass_constants( const Camera& camera, renderer::Extent2D extent, uint32_t sample_count, float lod_pixel_tolerance )
	{
		const glm::mat4 view = camera.calc_view_matrix();
		// https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
		const auto f = 1.0f / tan( camera.fov / 2.0f );
		const auto aspect_ratio = static_cast<float>( extent.width ) / extent.height;
		glm::mat4 projection( f / aspect_ratio, 0.0f, 0.0f, 0.0f, 0.0f, f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.01f, 0.0f );

		pass_constants constants { .view_projection_matrix = projection * view,
								   .projection_matrix = projection,
								   .view_matrix = view,
								   .inverse_view_matrix = glm::inverse( view ),
								   .view_position = camera.position,
								   .viewport_size = { extent.width, extent.height },
								   .sample_count = sample_count,
								   // World size covered by the tolerance at a distance of 1
								   .lod_target = lod_pixel_tolerance / ( extent.height / ( 2 * std::tan( camera.fov * 0.5f ) ) ) };

		// Math from https://github.com/zeux/niagara
		const float draw_distance = 500.f;
		const auto transposed_vp = glm::transpose( constants.view_projection_matrix );
		constants.frustum[ 0 ] = normalize_plane( transposed_vp[ 3 ] + transposed_vp[ 0 ] ); // x + w < 0
		constants.frustum[ 1 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 0 ] ); // x - w > 0
		constants.frustum[ 2 ] = normalize_plane( transposed_vp[ 3 ] + transposed_vp[ 1 ] ); // y + w < 0
		constants.frustum[ 3 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 1 ] ); // y - w > 0
		constants.frustum[ 4 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 2 ] ); // z - w > 0 -- reverse z
		constants.frustum[ 5 ] = glm::vec4( 0, 0, -1, draw_distance ); // reverse z, infinite far plane

		return constants;
	}

	renderer::BufferHandle calc_cascade_pass_constants_address( const renderer::Buffer& buffer, size_t cascade )
	{
		return buffer.get_device_address() + cascade * sizeof( pass_constants );
	}
}

template <typename Commands>
SceneRenderer::DrawCommandsSegments SceneRenderer::make_empty_draw_commands_segments()
{
	DrawCommandsSegments segments;
	segments.fill( { .commands_offset = offsetof( Commands, commands ), .capacity = 0 } );
	return segments;
}

template <typename Commands>
void SceneRenderer::add_draw_commands( DrawCommandsSegments& segments, size_t bucket, uint32_t count )
{
	using Command = typename decltype( Commands::commands )::value_type;
	segments[ bucket ].capacity += count;
	for ( size_t i = bucket + 1; i < DRAW_BUCKETS_COUNT; ++i )
	{
		segments[ i ].commands_offset += count * sizeof( Command );
	}
	assert( segments.back().commands_offset + segments.back().capacity * sizeof( Command ) <= sizeof( Commands ) );
}

using usage = renderer::Texture::Usage;

SceneRenderer::SceneRenderer( renderer::Device& device,
							  renderer::Texture::Format output_format,
							  int sample_count,
							  BindlessManager& bindless_manager,
							  renderer::PipelineManager& pipelines )
	: _sample_count( sample_count )
	, _extent( device.get_extent() )
	, _depth_pyramid_extent( round_pow2( _extent.width ), round_pow2( _extent.height ) )
	, _resolved_color_texture( bindless_manager.add_texture( device.create_texture(
		  renderer::Texture::Desc { .format = COLOR_FORMAT, .usage = usage::COLOR_ATTACHMENT | usage::SAMPLED, .extent = _extent } ) ) )
	, _resolved_depth_texture( bindless_manager.add_texture(
		  device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::D32_SFLOAT,
														   .usage = usage::DEPTH_STENCIL_ATTACHMENT | usage::SAMPLED,
														   .extent = _extent } ) ) )
	, _depth_pyramid( bindless_manager.add_texture(
		  device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::R32_SFLOAT,
														   .usage = usage::SAMPLED | usage::STORAGE,
														   .extent = _depth_pyramid_extent,
														   .mips = calc_mip_levels( _depth_pyramid_extent ) } ),
		  true ) )
	, _shadowmap( bindless_manager.add_texture(
		  device.create_texture( renderer::Texture::Desc { .format = SHADOW_DEPTH_FORMAT,
														   .usage = usage::DEPTH_STENCIL_ATTACHMENT | usage::SAMPLED,
														   .extent = { .width = 2048, .height = 2048 },
														   .layers = SHADOW_CASCADE_MAX } ) ) )
	, _frame_constant_buffers(
		  device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
								sizeof( std::array<frame_constants, renderer::MAX_FRAMES_IN_FLIGHT> ),
								renderer::Buffer::CpuUsage::WRITE ) )
	, _pass_constant_buffers(
		  device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
								sizeof( std::array<pass_constants, renderer::MAX_FRAMES_IN_FLIGHT> ),
								renderer::Buffer::CpuUsage::WRITE ) )
	, _shadow_cascades_buffer(
		  device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
								sizeof( gpu_shadow_cascades ) ) )
	, _shadowmap_pass_constant_buffers(
		  device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
								sizeof( std::array<pass_constants, SHADOW_CASCADE_MAX> ) ) )
	, _depth_range_buffer( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
													 | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
												 sizeof( gpu_depth_range ) ) )
	, _draw_buffers( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
										   sizeof( std::array<MeshDraw, MAX_DRAWS> ) * renderer::MAX_FRAMES_IN_FLIGHT,
										   renderer::Buffer::CpuUsage::WRITE ) )
	, _draw_visibility_buffer( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
														 | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
													 MAX_DRAWS ) )
	, _shadowmap_visibility_buffer( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
															  | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
														  MAX_DRAWS ) )
	, _meshlet_visibility_buffer(
		  device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
								MAX_MESHLET_VISIBILITY_BITS / 8 ) )
	, _draw_statistics_buffer( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_SRC
														 | renderer::Buffer::Usage::TRANSFER_DST
														 | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
													 sizeof( DrawStatistics ) ) )
	, _draw_indirect_count( device.get_properties().draw_indirect_count_support )
{
	_draw_mesh_segments = make_empty_draw_commands_segments<gpu_draw_mesh_commands>();
	_task_segments = make_empty_draw_commands_segments<gpu_task_commands>();

	const std::vector<renderer::ShaderSource::Define> mesh_shaders_defines { { "MESH_LOD_COUNT", std::to_string( MESH_LOD_COUNT ) },
																			 { "TASK_GROUP_SIZE", std::to_string( TASK_GROUP_SIZE ) },
																			 { "DRAW_BUCKETS_COUNT", std::to_string( DRAW_BUCKETS_COUNT ) },
																			 { "DRAW_BINS_PER_BUCKET",
																			   std::to_string( DRAW_BINS_PER_BUCKET ) },
																			 { "USE_ALPHA", "false" } };
	const std::vector<renderer::ShaderSource::Define> mesh_alpha_shaders_defines {
		{ "MESH_LOD_COUNT", std::to_string( MESH_LOD_COUNT ) },
		{ "TASK_GROUP_SIZE", std::to_string( TASK_GROUP_SIZE ) },
		{ "DRAW_BUCKETS_COUNT", std::to_string( DRAW_BUCKETS_COUNT ) },
		{ "DRAW_BINS_PER_BUCKET", std::to_string( DRAW_BINS_PER_BUCKET ) },
		{ "USE_ALPHA", "true" }
	};

	const auto mesh_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
								   .depth_format = DEPTH_FORMAT,
								   .sample_count = _sample_count,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines },
		  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

	const auto mesh_alpha_clip_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
								   .depth_format = DEPTH_FORMAT,
								   .sample_count = _sample_count,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines },
		  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_alpha_shaders_defines } } );

	const auto mesh_alpha_blend_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
								   .depth_format = DEPTH_FORMAT,
								   .depth_compare_op = renderer::CompareOp::GREATER_OR_EQUAL,
								   .sample_count = _sample_count,
								   .alpha_blend = true,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines },
		  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

	const auto shadowmap_mesh_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .depth_format = SHADOW_DEPTH_FORMAT,
								   .depth_compare_op = renderer::CompareOp::LESS,
								   .depth_clamp = true,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines } } );

	const auto shadowmap_mesh_alpha_clip_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .depth_format = SHADOW_DEPTH_FORMAT,
								   .depth_compare_op = renderer::CompareOp::LESS,
								   .depth_clamp = true,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines },
		  { .path = "shadow_alpha_test.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

	// Without mesh shaders, the steps keep invalid meshlet pipelines that draw_buckets() never reads
	renderer::PipelineHandle meshlet_pipeline;
	renderer::PipelineHandle meshlet_alpha_clip_pipeline;
	renderer::PipelineHandle meshlet_alpha_blend_pipeline;
	renderer::PipelineHandle shadowmap_meshlet_pipeline;
	renderer::PipelineHandle shadowmap_meshlet_alpha_clip_pipeline;
	if ( device.get_properties().mesh_shader_support )
	{
		meshlet_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
									   .depth_format = DEPTH_FORMAT,
									   .sample_count = _sample_count,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines },
			  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

		meshlet_alpha_clip_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
									   .depth_format = DEPTH_FORMAT,
									   .sample_count = _sample_count,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines },
			  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_alpha_shaders_defines } } );

		meshlet_alpha_blend_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
									   .depth_format = DEPTH_FORMAT,
									   .depth_compare_op = renderer::CompareOp::GREATER_OR_EQUAL,
									   .sample_count = _sample_count,
									   .alpha_blend = true,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines },
			  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

		shadowmap_meshlet_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .depth_format = SHADOW_DEPTH_FORMAT,
									   .depth_compare_op = renderer::CompareOp::LESS,
									   .depth_clamp = true,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines } } );

		shadowmap_meshlet_alpha_clip_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .depth_format = SHADOW_DEPTH_FORMAT,
									   .depth_compare_op = renderer::CompareOp::LESS,
									   .depth_clamp = true,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines },
			  { .path = "shadow_alpha_test.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );
	}

	_opaque_steps = { { { .label = "Opaque",
						  .meshlet_pipeline = meshlet_pipeline,
						  .mesh_pipeline = mesh_pipeline,
						  .first_bucket = material_type::OPAQUE_SINGLE },
						{ .label = "Alpha tested",
						  .meshlet_pipeline = meshlet_alpha_clip_pipeline,
						  .mesh_pipeline = mesh_alpha_clip_pipeline,
						  .first_bucket = material_type::MASK_SINGLE } } };
	_shadow_steps = { { { .label = "Opaque",
						  .meshlet_pipeline = shadowmap_meshlet_pipeline,
						  .mesh_pipeline = shadowmap_mesh_pipeline,
						  .first_bucket = material_type::OPAQUE_SINGLE },
						{ .label = "Alpha tested",
						  .meshlet_pipeline = shadowmap_meshlet_alpha_clip_pipeline,
						  .mesh_pipeline = shadowmap_mesh_alpha_clip_pipeline,
						  .first_bucket = material_type::MASK_SINGLE } } };
	_transparent_steps = { { { .label = "Clip",
							   .meshlet_pipeline = meshlet_alpha_clip_pipeline,
							   .mesh_pipeline = mesh_alpha_clip_pipeline,
							   .first_bucket = material_type::ALPHA_SINGLE },
							 { .label = "Blend",
							   .meshlet_pipeline = meshlet_alpha_blend_pipeline,
							   .mesh_pipeline = mesh_alpha_blend_pipeline,
							   .first_bucket = material_type::ALPHA_SINGLE,
							   .depth_write = false } } };

	std::vector<renderer::ShaderSource::Define> compute_shader_defines;
	compute_shader_defines.emplace_back( "LATE", "false" );
	compute_shader_defines.emplace_back( "TASK_COMMANDS", "0" ); // Used in #if, true/false would both be 0
	std::copy( begin( mesh_shaders_defines ), end( mesh_shaders_defines ), std::back_inserter( compute_shader_defines ) );

	const auto add_compute_draws_pipeline = [ & ]( bool late, bool task_commands )
	{
		compute_shader_defines[ 0 ].value = late ? "true" : "false";
		compute_shader_defines[ 1 ].value = task_commands ? "1" : "0";
		return pipelines.add(
			renderer::Pipeline::Desc { .push_constants_size = sizeof( compute_draws_constants ) },
			{ { .path = "mesh_cull.comp", .stage = renderer::ShaderStage::COMPUTE, .defines = compute_shader_defines } } );
	};
	_compute_draws_early_pipeline = add_compute_draws_pipeline( false, false );
	_compute_draws_late_pipeline = add_compute_draws_pipeline( true, false );
	_compute_tasks_early_pipeline = add_compute_draws_pipeline( false, true );
	_compute_tasks_late_pipeline = add_compute_draws_pipeline( true, true );

	const auto add_draw_commands_pipeline = [ & ]( bool task_commands )
	{
		compute_shader_defines[ 1 ].value = task_commands ? "1" : "0";
		return pipelines.add(
			renderer::Pipeline::Desc { .push_constants_size = sizeof( draw_commands_constants ) },
			{ { .path = "mesh_commands.comp", .stage = renderer::ShaderStage::COMPUTE, .defines = compute_shader_defines } } );
	};
	_draw_commands_pipeline = add_draw_commands_pipeline( false );
	_task_commands_pipeline = add_draw_commands_pipeline( true );

	_depth_pyramid_pipeline = pipelines.add( renderer::Pipeline::Desc { .push_constants_size = sizeof( depth_pyramid_constants ) },
											 { { .path = "depth_pyramid.comp", .stage = renderer::ShaderStage::COMPUTE } } );
	_depth_range_pipeline = pipelines.add( renderer::Pipeline::Desc { .push_constants_size = sizeof( depth_range_constants ) },
										   { { .path = "depth_range.comp", .stage = renderer::ShaderStage::COMPUTE } } );
	_shadow_cascades_pipeline = pipelines.add( renderer::Pipeline::Desc { .push_constants_size = sizeof( shadow_cascades_constants ) },
											   { { .path = "shadow_cascades.comp", .stage = renderer::ShaderStage::COMPUTE } } );

	_grid_pipeline = pipelines.add( renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
															   .depth_format = DEPTH_FORMAT,
															   .topology = renderer::Pipeline::PrimitiveTopology::TRIANGLE_STRIP,
															   .sample_count = _sample_count,
															   .push_constants_size = sizeof( grid_constants ) },
									{ { .path = "grid.vert", .stage = renderer::ShaderStage::VERTEX },
									  { .path = "grid.frag", .stage = renderer::ShaderStage::FRAGMENT } } );

	_tonemap_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = output_format, .push_constants_size = sizeof( renderer::BindlessSampledTexture ) },
		{ { .path = "fullscreen.vert", .stage = renderer::ShaderStage::VERTEX },
		  { .path = "tonemap.frag", .stage = renderer::ShaderStage::FRAGMENT } } );

	const auto create_draw_commands_buffer = [ & ]()
	{
		return device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::INDIRECT_BUFFER
										 | renderer::Buffer::Usage::TRANSFER_DST | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
									 sizeof( gpu_draw_commands_buffer ) );
	};
	_draw_commands_buffer = create_draw_commands_buffer();
	for ( auto& buffer : _shadowmap_draw_commands_buffers )
	{
		buffer = create_draw_commands_buffer();
	}

	for ( auto& buffer : _draw_statistics_readback_buffers )
	{
		buffer = device.create_buffer( renderer::Buffer::Usage::TRANSFER_DST,
									   sizeof( DrawStatistics ),
									   renderer::Buffer::CpuUsage::READ_WRITE );
	}

	for ( size_t layer = 0; layer < SHADOW_CASCADE_MAX; ++layer )
	{
		_shadowmap_layer_views[ layer ] = device.create_texture_view( _shadowmap.texture,
																	  renderer::TextureView::Aspect::DEPTH,
																	  -1,
																	  static_cast<int>( layer ) );
	}

	// With a single sample, passes render straight into the resolved textures
	if ( _sample_count > 1 )
	{
		_color_texture = device.create_texture( renderer::Texture::Desc { .format = COLOR_FORMAT,
																		  .usage = usage::COLOR_ATTACHMENT,
																		  .extent = _extent,
																		  .samples = _sample_count } );
		_color_texture_view = device.create_texture_view( _color_texture, renderer::TextureView::Aspect::COLOR );
		_depth_texture = device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::D32_SFLOAT,
																		  .usage = usage::DEPTH_STENCIL_ATTACHMENT,
																		  .extent = _extent,
																		  .samples = _sample_count } );
		_depth_texture_view = device.create_texture_view( _depth_texture, renderer::TextureView::Aspect::DEPTH );
	}

	auto cmd = device.grab_command_buffer();
	cmd->reset();
	cmd->begin();
	const auto init_texture = [ & ]( const renderer::Texture& texture )
	{ cmd->transition_texture( texture, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL ); };
	init_texture( _resolved_color_texture.texture );
	init_texture( _resolved_depth_texture.texture );
	init_texture( _depth_pyramid.texture );
	init_texture( _shadowmap.texture );
	if ( _sample_count > 1 )
	{
		init_texture( _color_texture );
		init_texture( _depth_texture );
	}
	// Shadow passes don't do occlusion culling, they draw everything
	cmd->fill_buffer( _shadowmap_visibility_buffer, 0, _shadowmap_visibility_buffer.get_size(), 0x01010101 );
	// Nothing measured yet, the first cascades cover the whole shadow distance
	cmd->fill_buffer( _depth_range_buffer, 0, _depth_range_buffer.get_size(), 0 );
	cmd->barrier( renderer::Stage::TRANSFER, renderer::Stage::ALL );
	cmd->end();
	device.submit_and_wait( *cmd );
}

void SceneRenderer::add_mesh( const mesh& mesh, glm::vec3 translation, glm::quat rotation, glm::vec3 scale, const material& material )
{
	assert( _scene.size() < MAX_DRAWS );
	assert( _meshlet_visibility_bits + mesh.meshlets_count <= MAX_MESHLET_VISIBILITY_BITS );
	assert( std::accumulate( begin( _task_segments ),
							 end( _task_segments ),
							 0u,
							 []( uint32_t total, const DrawCommandsSegment& segment ) { return total + segment.capacity; } )
				+ mesh.max_task_groups
			<= MAX_TASK_COMMANDS );

	_scene.push_back( MeshDraw { .translation = translation,
								 .mesh = mesh.handle,
								 .rotation = rotation,
								 .scale = scale,
								 .material = material.handle,
								 .meshlet_visibility_offset = _meshlet_visibility_bits } );
	_meshlet_visibility_bits += mesh.meshlets_count;
	const auto bucket = std::to_underlying( material.type );
	add_draw_commands<gpu_draw_mesh_commands>( _draw_mesh_segments, bucket, 1 );
	add_draw_commands<gpu_task_commands>( _task_segments, bucket, mesh.max_task_groups );
	_needs_rebuild = true;
}

void SceneRenderer::clear_meshes()
{
	_scene.clear();
	_meshlet_visibility_bits = 0;
	_draw_mesh_segments = make_empty_draw_commands_segments<gpu_draw_mesh_commands>();
	_task_segments = make_empty_draw_commands_segments<gpu_task_commands>();
	_needs_rebuild = true;
}

void SceneRenderer::draw_bin( renderer::CommandBuffer& cmd, const renderer::Buffer& draw_commands, size_t bin ) const
{
	if ( _task_segments[ bin / DRAW_BINS_PER_BUCKET ].capacity == 0 )
	{
		return;
	}

	// All the task workgroups in a single dispatch, a group count of 0 draws nothing
	const auto header_offset = calc_header_offset<gpu_task_commands>( bin );
	cmd.draw_mesh_tasks_indirect( draw_commands,
								  header_offset + offsetof( gpu_task_header, dispatch ),
								  draw_commands,
								  header_offset + offsetof( gpu_task_header, draw_count ),
								  1,
								  sizeof( VkDrawMeshTasksIndirectCommandEXT ) );
}

void SceneRenderer::draw_bucket( renderer::CommandBuffer& cmd, const renderer::Buffer& draw_commands, size_t bucket ) const
{
	const auto& segment = _draw_mesh_segments[ bucket ];
	if ( segment.capacity == 0 )
	{
		return;
	}

	if ( _draw_indirect_count )
	{
		cmd.draw_indexed_indirect( draw_commands,
								   segment.commands_offset + offsetof( gpu_draw_mesh_command, draw_cmd ),
								   draw_commands,
								   offsetof( gpu_draw_mesh_commands, bucket_counts ) + bucket * sizeof( uint32_t ),
								   segment.capacity,
								   sizeof( gpu_draw_mesh_command ) );
	}
	else
	{
		// The commands past the count are cleared, they draw nothing
		cmd.draw_indexed_indirect( draw_commands,
								   segment.commands_offset + offsetof( gpu_draw_mesh_command, draw_cmd ),
								   segment.capacity,
								   sizeof( gpu_draw_mesh_command ) );
	}
}

void SceneRenderer::render( const BindlessManager& bindless_manager,
							const renderer::PipelineManager& pipelines,
							renderer::CommandBuffer& cmd,
							renderer::Texture output_texture,
							renderer::TextureView output_texture_view,
							const renderer::Buffer* index_buffer,
							renderer::TimestampQuery timings_pool,
							const Camera& camera,
							glm::vec4 main_light,
							float sky_illumination,
							float lod_pixel_tolerance,
							float shadow_distance,
							uint32_t shadow_cascade_count,
							bool show_grid,
							bool show_shadow_cascades,
							bool use_pbr,
							bool use_occlusion_culling )
{
	const auto frame_index = _frames % renderer::MAX_FRAMES_IN_FLIGHT;

	if ( _frames >= renderer::MAX_FRAMES_IN_FLIGHT )
	{
		_draw_statistics_readback_buffers[ frame_index ].flush();
		memcpy( &_statistics, _draw_statistics_readback_buffers[ frame_index ].get_mapped_address(), sizeof( DrawStatistics ) );
	}

	if ( index_buffer )
	{
		cmd.bind_index_buffer( *index_buffer );
	}

	using Draws = std::array<MeshDraw, MAX_DRAWS>;
	memcpy( static_cast<Draws*>( _draw_buffers.get_mapped_address() ) + frame_index, _scene.data(), sizeof( MeshDraw ) * _scene.size() );
	const auto draw_buffer_address = _draw_buffers.get_device_address() + ( frame_index * sizeof( Draws ) );

	const bool use_mesh_shader = index_buffer == nullptr;
	const uint32_t cascade_count = std::min( shadow_cascade_count, SHADOW_CASCADE_MAX );

	{
		const glm::mat4 inverse_view = glm::inverse( camera.calc_view_matrix() );
		frame_constants constants { .camera_position = camera.position,
									.camera_forward = -glm::vec3( inverse_view[ 2 ] ),
									.main_light = main_light,
									.sky_to_sun = sky_illumination,
									.pbr = use_pbr ? 1u : 0u };
		if ( cascade_count > 0 )
		{
			constants.shadow_map = _shadowmap.handles.sampled_texture;
			constants.shadow_cascade_count = cascade_count;
			constants.shadow_distance = shadow_distance;
			constants.shadow_cascades = _shadow_cascades_buffer.get_device_address();
			constants.show_shadow_cascades = show_shadow_cascades ? 1 : 0;
		}
		memcpy( static_cast<frame_constants*>( _frame_constant_buffers.get_mapped_address() ) + frame_index,
				&constants,
				sizeof( constants ) );
	}
	const auto frame_constants_address = _frame_constant_buffers.get_device_address() + ( frame_index * sizeof( frame_constants ) );

	{
		const auto constants = build_main_pass_constants( camera, _extent, static_cast<uint32_t>( _sample_count ), lod_pixel_tolerance );
		memcpy( static_cast<pass_constants*>( _pass_constant_buffers.get_mapped_address() ) + frame_index,
				&constants,
				sizeof( constants ) );
	}
	const auto pass_constants_address = _pass_constant_buffers.get_device_address() + ( frame_index * sizeof( pass_constants ) );

	// Previous frame's shading and shadow passes read the cascades, its culling read the visibility, it measured the
	// depth range and copied the statistics to its readback buffer
	cmd.barrier( renderer::Stage::ANY_SHADER | renderer::Stage::TRANSFER, renderer::Stage::COMPUTE | renderer::Stage::TRANSFER );
	if ( cascade_count > 0 )
	{
		setup_shadow_cascades( bindless_manager, pipelines, cmd, frame_constants_address, pass_constants_address );
		// Shadow passes and shading read the cascades, the depth range gets reset
		cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::ANY_SHADER | renderer::Stage::TRANSFER );
		cmd.fill_buffer( _depth_range_buffer, offsetof( gpu_depth_range, nearest ), sizeof( uint32_t ), 0 );
		cmd.fill_buffer( _depth_range_buffer, offsetof( gpu_depth_range, farthest ), sizeof( uint32_t ), 0xffffffff );
	}
	// Can't reuse the visibility buffer if the scene has changed. Without occlusion culling it forces everything to be
	// drawn in the early pass, which only needs to be filled once
	if ( use_occlusion_culling ? _needs_rebuild : !_all_visible )
	{
		cmd.fill_buffer( _draw_visibility_buffer, 0, _draw_visibility_buffer.get_size(), use_occlusion_culling ? 0 : 0x01010101 );
	}
	_needs_rebuild = !use_occlusion_culling;
	_all_visible = !use_occlusion_culling;
	cmd.fill_buffer( _draw_statistics_buffer, 0, sizeof( DrawStatistics ), 0 );
	cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::COMPUTE );

	if ( cascade_count > 0 )
	{
		const auto label = cmd.debug_label( "Shadow Map" );
		draw_shadowmap( bindless_manager,
						pipelines,
						cmd,
						timings_pool,
						frame_constants_address,
						draw_buffer_address,
						_draw_statistics_buffer.get_device_address(),
						cascade_count,
						use_mesh_shader );
	}

	const bool meshlet_occlusion = use_occlusion_culling && use_mesh_shader;
	if ( meshlet_occlusion )
	{
		// Previous frame's late pass wrote the bits the early pass reads
		cmd.barrier( renderer::Stage::VERTEX, renderer::Stage::VERTEX );
	}

	// The main passes render into the MSAA targets, then resolve. With a single sample they render straight into the
	// resolved textures
	const bool msaa = _sample_count > 1;
	const renderer::TextureView color_target_view = msaa ? _color_texture_view : _resolved_color_texture.handles.view;
	const renderer::TextureView depth_target_view = msaa ? _depth_texture_view : _resolved_depth_texture.handles.view;

	{
		const auto label = cmd.debug_label( "Opaque Early" );
		renderer::RenderInfo render_info { .color_target = { .target = color_target_view, .clear_value = { { 0.1f, 0.1f, 0.1f, 1.f } } },
										   .depth_target = { .target = depth_target_view, .clear_value = { { 0.f } } } };
		{
			const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::CULLING_EARLY );
			// Last frame's targets, the resolved color tonemapping read and the resolved depth the depth pyramid read
			cmd.barrier( renderer::Stage::COLOR | renderer::Stage::DEPTH | renderer::Stage::FRAGMENT | renderer::Stage::COMPUTE,
						 renderer::Stage::COLOR | renderer::Stage::DEPTH );
			// The depth pyramid reads the early pass depth, the shadow depth range the final depth
			if ( msaa && ( use_occlusion_culling || cascade_count > 0 ) )
			{
				render_info.depth_target.resolve_target = _resolved_depth_texture.handles.view;
				// Both keep the farthest depth, which is the minimum with reverse-Z
				render_info.depth_target.resolve_mode = renderer::RenderAttachment::ResolveMode::MIN;
			}
			if ( msaa && !use_occlusion_culling )
			{
				// Without occlusion culling we draw all in one pass so it needs to resolve at the end
				render_info.color_target.resolve_target = _resolved_color_texture.handles.view;
			}

			compute_draws( bindless_manager,
						   pipelines,
						   cmd,
						   pass_constants_address,
						   draw_buffer_address,
						   _draw_visibility_buffer,
						   _draw_statistics_buffer.get_device_address(),
						   _draw_commands_buffer,
						   true,
						   meshlet_occlusion,
						   use_mesh_shader );
		}

		const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::DRAW_OPAQUES_EARLY );
		cmd.begin_rendering( render_info );
		draw_opaques( bindless_manager,
					  pipelines,
					  cmd,
					  frame_constants_address,
					  pass_constants_address,
					  draw_buffer_address,
					  use_mesh_shader,
					  meshlet_occlusion ? MeshletPass::EARLY : MeshletPass::NONE );
		// Do not end rendering, the later work decides if we continue on this one or not
	}


	if ( use_occlusion_culling )
	{
		cmd.end_rendering();
		// The depth pyramid reads the resolved depth, and overwrites what last frame's culling and task shaders read
		cmd.barrier( renderer::Stage::COLOR | renderer::Stage::DEPTH | renderer::Stage::VERTEX, renderer::Stage::COMPUTE );

		{
			PROFILER_GPU_EVENT( "Depth Pyramid" );
			const auto label = cmd.debug_label( "Depth Pyramid" );
			const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::DEPTH_PYRAMID );
			cmd.bind_pipeline( pipelines.get( _depth_pyramid_pipeline ), bindless_manager );
			auto extent = _depth_pyramid_extent;
			for ( size_t mip = 0; mip < _depth_pyramid.mips.size(); ++mip )
			{
				const depth_pyramid_constants constants { .depth_in = mip == 0 ? _resolved_depth_texture.handles.sampled_texture
																			   : _depth_pyramid.mips[ mip - 1 ].sampled_texture,
														  .depth_out = _depth_pyramid.mips[ mip ].storage_texture,
														  .size = glm::aligned_vec2( extent.width, extent.height ) };
				cmd.push_constants( pipelines.get( _depth_pyramid_pipeline ), constants );
				cmd.dispatch( calc_group_count( extent.width ), calc_group_count( extent.height ), 1 );

				cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::COMPUTE );

				extent.width = std::max( extent.width / 2, 1u );
				extent.height = std::max( extent.height / 2, 1u );
			}
		}

		const auto label = cmd.debug_label( "Opaque Late" );
		{
			const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::CULLING_LATE );
			compute_draws( bindless_manager,
						   pipelines,
						   cmd,
						   pass_constants_address,
						   draw_buffer_address,
						   _draw_visibility_buffer,
						   _draw_statistics_buffer.get_device_address(),
						   _draw_commands_buffer,
						   false,
						   meshlet_occlusion,
						   use_mesh_shader );
		}

		const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::DRAW_OPAQUES_LATE );
		if ( meshlet_occlusion )
		{
			cmd.barrier( renderer::Stage::VERTEX, renderer::Stage::VERTEX );
		}

		renderer::RenderInfo render_info { .color_target = { .target = color_target_view },
										   .depth_target = { .target = depth_target_view } };
		// The depth pyramid read the depth, which this pass writes with a single sample, or resolves into with MSAA
		cmd.barrier( renderer::Stage::COMPUTE, msaa ? renderer::Stage::COLOR : renderer::Stage::DEPTH );
		if ( msaa )
		{
			render_info.color_target.resolve_target = _resolved_color_texture.handles.view;
			if ( cascade_count > 0 )
			{
				render_info.depth_target.resolve_target = _resolved_depth_texture.handles.view;
				render_info.depth_target.resolve_mode = renderer::RenderAttachment::ResolveMode::MIN;
			}
		}
		cmd.begin_rendering( render_info );

		draw_opaques( bindless_manager,
					  pipelines,
					  cmd,
					  frame_constants_address,
					  pass_constants_address,
					  draw_buffer_address,
					  use_mesh_shader,
					  meshlet_occlusion ? MeshletPass::LATE : MeshletPass::NONE );
	}

	{
		const auto label = cmd.debug_label( "Transparents" );
		const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::DRAW_TRANSPARENTS );
		PROFILER_GPU_EVENT( "Draw Transparents" );
		draw_buckets(
			bindless_manager,
			pipelines,
			cmd,
			{ .draw_commands = _draw_commands_buffer, .viewport = _extent, .use_mesh_shader = use_mesh_shader },
			_transparent_steps,
			{ .frame_constants = frame_constants_address, .pass_constants = pass_constants_address, .draws_buffer = draw_buffer_address } );
	}

	{
		PROFILER_GPU_EVENT( "Draw Grid" );
		const auto label = cmd.debug_label( "Grid" );
		const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::DRAW_GRID );

		if ( show_grid )
		{
			cmd.set_dynamic_state( { .viewport = _extent } );
			cmd.bind_pipeline( pipelines.get( _grid_pipeline ), bindless_manager );
			cmd.push_constants( pipelines.get( _grid_pipeline ),
								grid_constants { .frame_constants = frame_constants_address, .pass_constants = pass_constants_address } );
			cmd.draw( 4 );
		}
	}

	cmd.end_rendering();

	if ( cascade_count > 0 )
	{
		PROFILER_GPU_EVENT( "Depth Range" );
		const auto label = cmd.debug_label( "Depth Range" );
		// Measured after transparents and the grid: they write depth too, and receive shadows
		cmd.barrier( renderer::Stage::DEPTH | renderer::Stage::COLOR, renderer::Stage::COMPUTE );
		const auto& pipeline = pipelines.get( _depth_range_pipeline );
		cmd.bind_pipeline( pipeline, bindless_manager );
		cmd.push_constants( pipeline,
							depth_range_constants { .depth_range = _depth_range_buffer.get_device_address(),
													.depth = _resolved_depth_texture.handles.sampled_texture } );
		// Each workgroup reduces 64x64 pixels
		cmd.dispatch( ( _extent.width + 63 ) / 64, ( _extent.height + 63 ) / 64, 1 );
	}

	cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::TRANSFER );
	cmd.copy_buffer( _draw_statistics_buffer, 0, sizeof( DrawStatistics ), _draw_statistics_readback_buffers[ frame_index ] );
	cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::HOST );

	{
		PROFILER_GPU_EVENT( "Tonemap" );
		const auto label = cmd.debug_label( "Tonemap" );
		const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::TONEMAP );
		cmd.barrier( renderer::Stage::COLOR, renderer::Stage::FRAGMENT );
		cmd.transition_texture( output_texture, renderer::Texture::Layout::PRESENT_SRC, renderer::Texture::Layout::GENERAL );
		cmd.begin_rendering( { .color_target = { .target = output_texture_view } } );
		cmd.set_dynamic_state( { .viewport = output_texture.get_extent(), .depth_test = false, .depth_write = false } );
		cmd.bind_pipeline( pipelines.get( _tonemap_pipeline ), bindless_manager );
		cmd.push_constants( pipelines.get( _tonemap_pipeline ), _resolved_color_texture.handles.sampled_texture );
		cmd.draw( 3 );
		cmd.end_rendering();
	}

	++_frames;
}

void SceneRenderer::compute_draws( const BindlessManager& bindless_manager,
								   const renderer::PipelineManager& pipelines,
								   renderer::CommandBuffer& cmd,
								   renderer::BufferHandle pass_constants_address,
								   renderer::BufferHandle draw_buffer_address,
								   const renderer::Buffer& visibility_buffer,
								   renderer::BufferHandle stats_buffer_address,
								   const renderer::Buffer& draw_commands,
								   bool early,
								   bool meshlet_occlusion,
								   bool task_commands ) const
{
	PROFILER_GPU_EVENT( "Compute Draws" );
	const auto label = cmd.debug_label( "Culling" );
	// Previous draws read commands both as indirect arguments and from shaders
	cmd.barrier( renderer::Stage::INDIRECT | renderer::Stage::VERTEX, renderer::Stage::TRANSFER );
	clear_draw_commands( cmd, draw_commands, task_commands );
	cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::COMPUTE );
	dispatch_culling( bindless_manager,
					  pipelines,
					  cmd,
					  pass_constants_address,
					  draw_buffer_address,
					  visibility_buffer,
					  stats_buffer_address,
					  draw_commands,
					  early,
					  meshlet_occlusion,
					  task_commands );
	cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::COMPUTE );
	dispatch_draw_commands( bindless_manager, pipelines, cmd, pass_constants_address, draw_buffer_address, draw_commands, task_commands );
	// Task and vertex shaders also read the commands, the next culling reads the visibility
	cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::INDIRECT | renderer::Stage::VERTEX | renderer::Stage::COMPUTE );
}

void SceneRenderer::clear_draw_commands( renderer::CommandBuffer& cmd, const renderer::Buffer& draw_commands, bool task_commands ) const
{
	// Headers count from 0, mesh_commands.comp writes the rest of the task headers
	if ( task_commands )
	{
		cmd.fill_buffer( draw_commands, 0, offsetof( gpu_task_commands, commands ), 0 );
	}
	else
	{
		// Draws without draw indirect count support read all the buckets' commands, commands past the count must be empty
		const size_t commands_size = _draw_indirect_count ? 0 : _scene.size() * sizeof( gpu_draw_mesh_command );
		cmd.fill_buffer( draw_commands, 0, offsetof( gpu_draw_mesh_commands, commands ) + commands_size, 0 );
	}
}

void SceneRenderer::dispatch_culling( const BindlessManager& bindless_manager,
									  const renderer::PipelineManager& pipelines,
									  renderer::CommandBuffer& cmd,
									  renderer::BufferHandle pass_constants_address,
									  renderer::BufferHandle draw_buffer_address,
									  const renderer::Buffer& visibility_buffer,
									  renderer::BufferHandle stats_buffer_address,
									  const renderer::Buffer& draw_commands,
									  bool early,
									  bool meshlet_occlusion,
									  bool task_commands ) const
{
	compute_draws_constants constants { .draws_buffer = draw_buffer_address,
										.commands_buffer = draw_commands.get_device_address(),
										.slots_buffer = draw_commands.get_device_address() + offsetof( gpu_draw_commands_buffer, slots ),
										.visibility_buffer = visibility_buffer.get_device_address(),
										.statistics_buffer = stats_buffer_address,
										.pass_constants = pass_constants_address,
										.draws_count = static_cast<uint32_t>( _scene.size() ),
										.meshlet_occlusion = meshlet_occlusion ? 1u : 0u };

	if ( !early )
	{
		constants.depth_pyramid = _depth_pyramid.handles.sampled_texture;
	}

	const auto pipeline_handle = task_commands ? ( early ? _compute_tasks_early_pipeline : _compute_tasks_late_pipeline )
											   : ( early ? _compute_draws_early_pipeline : _compute_draws_late_pipeline );
	const auto pipeline = pipelines.get( pipeline_handle );
	cmd.bind_pipeline( pipeline, bindless_manager );

	cmd.push_constants( pipeline, constants );
	cmd.dispatch( calc_group_count( static_cast<uint32_t>( _scene.size() ) ), 1, 1 );
}

void SceneRenderer::dispatch_draw_commands( const BindlessManager& bindless_manager,
											const renderer::PipelineManager& pipelines,
											renderer::CommandBuffer& cmd,
											renderer::BufferHandle pass_constants_address,
											renderer::BufferHandle draw_buffer_address,
											const renderer::Buffer& draw_commands,
											bool task_commands ) const
{
	draw_commands_constants constants { .draws_buffer = draw_buffer_address,
										.commands_buffer = draw_commands.get_device_address(),
										.slots_buffer = draw_commands.get_device_address() + offsetof( gpu_draw_commands_buffer, slots ),
										.pass_constants = pass_constants_address,
										.draws_count = static_cast<uint32_t>( _scene.size() ) };
	for ( size_t i = 0; i < DRAW_BUCKETS_COUNT; ++i )
	{
		const auto commands_offset = get_draw_commands_segments( task_commands )[ i ].commands_offset;
		constants.bucket_first_commands[ i ] = task_commands ? calc_first_command<gpu_task_commands>( commands_offset )
															 : calc_first_command<gpu_draw_mesh_commands>( commands_offset );
	}

	const auto pipeline = pipelines.get( task_commands ? _task_commands_pipeline : _draw_commands_pipeline );
	cmd.bind_pipeline( pipeline, bindless_manager );
	cmd.push_constants( pipeline, constants );
	cmd.dispatch( calc_group_count( static_cast<uint32_t>( _scene.size() ) ), 1, 1 );
}

void SceneRenderer::draw_opaques( const BindlessManager& bindless_manager,
								  const renderer::PipelineManager& pipelines,
								  renderer::CommandBuffer& cmd,
								  renderer::BufferHandle frame_constants_address,
								  renderer::BufferHandle pass_constants_address,
								  renderer::BufferHandle draw_buffer_address,
								  bool use_mesh_shader,
								  MeshletPass meshlet_pass ) const
{
	PROFILER_GPU_EVENT( "Draw Opaques" );
	const auto label = cmd.debug_label( "Draw" );
	draw_buckets( bindless_manager,
				  pipelines,
				  cmd,
				  { .draw_commands = _draw_commands_buffer, .viewport = _extent, .use_mesh_shader = use_mesh_shader, .lod_labels = true },
				  _opaque_steps,
				  { .frame_constants = frame_constants_address,
					.pass_constants = pass_constants_address,
					.draws_buffer = draw_buffer_address,
					.meshlet_visibility = _meshlet_visibility_buffer.get_device_address(),
					.depth_pyramid = _depth_pyramid.handles.sampled_texture,
					.meshlet_pass = std::to_underlying( meshlet_pass ) } );
}

void SceneRenderer::draw_buckets( const BindlessManager& bindless_manager,
								  const renderer::PipelineManager& pipelines,
								  renderer::CommandBuffer& cmd,
								  const BucketDrawPass& pass,
								  const BucketDrawSteps& steps,
								  draw_meshes_constants constants ) const
{
	static constexpr std::array<const char*, DRAW_BINS_PER_BUCKET> bin_labels { "LOD 0", "LOD 1", "LOD 2", "LOD 3",
																				"LOD 4", "LOD 5", "LOD 6", "LOD 7" };
	for ( const auto& step : steps )
	{
		const auto step_label = cmd.debug_label( step.label );
		const auto& pipeline = pipelines.get( pass.use_mesh_shader ? step.meshlet_pipeline : step.mesh_pipeline );
		cmd.bind_pipeline( pipeline, bindless_manager );
		const auto set_dynamic_state = [ & ]( int index_in_pair )
		{
			cmd.set_dynamic_state( { .viewport = pass.viewport,
									 .cull_mode = bucket_cull_mode( index_in_pair ),
									 .depth_write = step.depth_write,
									 .depth_bias_constant = pass.depth_bias_constant,
									 .depth_bias_slope = pass.depth_bias_slope } );
		};

		if ( pass.use_mesh_shader )
		{
			constants.commands_buffer = pass.draw_commands.get_device_address();
			const auto draw_lod = [ & ]( size_t bin_in_bucket )
			{
				for ( int i = 0; i < 2; ++i )
				{
					const auto bin = calc_draw_bin( std::to_underlying( step.first_bucket ) + i, bin_in_bucket );
					set_dynamic_state( i );
					constants.draw_bin = static_cast<uint32_t>( bin );
					cmd.push_constants( pipeline, constants );
					draw_bin( cmd, pass.draw_commands, bin );
				}
			};
			// Bins in order: the nearest draws fill depth before the ones they hide, the sort key is the LOD
			for ( size_t bin_in_bucket = 0; bin_in_bucket < DRAW_BINS_PER_BUCKET; ++bin_in_bucket )
			{
				if ( pass.lod_labels )
				{
					const auto bin_label = cmd.debug_label( bin_labels[ bin_in_bucket ] );
					draw_lod( bin_in_bucket );
				}
				else
				{
					draw_lod( bin_in_bucket );
				}
			}
		}
		else
		{
			for ( int i = 0; i < 2; ++i )
			{
				const auto bucket = std::to_underlying( step.first_bucket ) + i;
				set_dynamic_state( i );
				constants.commands_buffer = pass.draw_commands.get_device_address() + _draw_mesh_segments[ bucket ].commands_offset;
				cmd.push_constants( pipeline, constants );
				draw_bucket( cmd, pass.draw_commands, bucket );
			}
		}
	}
}

void SceneRenderer::setup_shadow_cascades( const BindlessManager& bindless_manager,
										   const renderer::PipelineManager& pipelines,
										   renderer::CommandBuffer& cmd,
										   renderer::BufferHandle frame_constants_address,
										   renderer::BufferHandle pass_constants_address ) const
{
	PROFILER_GPU_EVENT( "Shadow Cascades" );
	const auto label = cmd.debug_label( "Shadow Cascades" );
	const shadow_cascades_constants constants { .frame_constants = frame_constants_address,
												.camera_pass_constants = pass_constants_address,
												.depth_range = _depth_range_buffer.get_device_address(),
												.shadow_cascades = _shadow_cascades_buffer.get_device_address(),
												.cascades_pass_constants = _shadowmap_pass_constant_buffers.get_device_address(),
												.shadowmap_resolution = _shadowmap.texture.get_extent().width };
	const auto& pipeline = pipelines.get( _shadow_cascades_pipeline );
	cmd.bind_pipeline( pipeline, bindless_manager );
	cmd.push_constants( pipeline, constants );
	cmd.dispatch( 1, 1, 1 );
}

void SceneRenderer::draw_shadowmap( const BindlessManager& bindless_manager,
									const renderer::PipelineManager& pipelines,
									renderer::CommandBuffer& cmd,
									renderer::TimestampQuery timings_pool,
									renderer::BufferHandle frame_constants_address,
									renderer::BufferHandle draw_buffer_address,
									renderer::BufferHandle stats_buffer_address,
									uint32_t cascade_count,
									bool use_mesh_shader )
{
	const auto calc_pass_constants_address = [ & ]( size_t cascade )
	{ return calc_cascade_pass_constants_address( _shadowmap_pass_constant_buffers, cascade ); };
	const std::span cascades_draw_commands( begin( _shadowmap_draw_commands_buffers ), cascade_count );

	// Each cascade has its own commands so that all the culling dispatches can run concurrently, then all the draws
	{
		PROFILER_GPU_EVENT( "Shadows Culling" );
		const auto label = cmd.debug_label( "Culling" );
		const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::CULLING_SHADOWS );
		// Previous frame's draws read commands both as indirect arguments and from shaders
		cmd.barrier( renderer::Stage::INDIRECT | renderer::Stage::VERTEX, renderer::Stage::TRANSFER );
		for ( const auto& draw_commands : cascades_draw_commands )
		{
			clear_draw_commands( cmd, draw_commands, use_mesh_shader );
		}
		cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::COMPUTE );
		// Shadow passes only read the visibility buffer, the dispatches don't depend on each other
		for ( size_t cascade = 0; cascade < cascade_count; ++cascade )
		{
			dispatch_culling( bindless_manager,
							  pipelines,
							  cmd,
							  calc_pass_constants_address( cascade ),
							  draw_buffer_address,
							  _shadowmap_visibility_buffer,
							  stats_buffer_address,
							  cascades_draw_commands[ cascade ],
							  true,
							  false,
							  use_mesh_shader );
		}
		cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::COMPUTE );
		for ( size_t cascade = 0; cascade < cascade_count; ++cascade )
		{
			dispatch_draw_commands( bindless_manager,
									pipelines,
									cmd,
									calc_pass_constants_address( cascade ),
									draw_buffer_address,
									cascades_draw_commands[ cascade ],
									use_mesh_shader );
		}
		// Task and vertex shaders also read the commands
		cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::INDIRECT | renderer::Stage::VERTEX );
	}

	PROFILER_GPU_EVENT( "Draw Shadows" );
	const auto label = cmd.debug_label( "Draw" );
	const GpuTimingScope timings_scope( cmd, timings_pool, GpuTiming::DRAW_SHADOWS );
	// Last frame's main pass sampled the shadow map
	cmd.barrier( renderer::Stage::FRAGMENT, renderer::Stage::DEPTH );
	static constexpr std::array<const char*, SHADOW_CASCADE_MAX> cascade_labels { "Cascade 0", "Cascade 1", "Cascade 2", "Cascade 3" };
	for ( size_t cascade = 0; cascade < cascade_count; ++cascade )
	{
		const auto cascade_label = cmd.debug_label( cascade_labels[ cascade ] );
		const auto& draw_commands = cascades_draw_commands[ cascade ];
		cmd.begin_rendering( { .depth_target = { .target = _shadowmap_layer_views[ cascade ], .clear_value = { { 1.f } } } } );
		draw_buckets( bindless_manager,
					  pipelines,
					  cmd,
					  { .draw_commands = draw_commands,
						.viewport = _shadowmap.texture.get_extent(),
						.depth_bias_constant = SHADOW_DEPTH_BIAS_CONSTANT,
						.depth_bias_slope = SHADOW_DEPTH_BIAS_SLOPE,
						.use_mesh_shader = use_mesh_shader },
					  _shadow_steps,
					  { .frame_constants = frame_constants_address,
						.pass_constants = calc_pass_constants_address( cascade ),
						.draws_buffer = draw_buffer_address } );
		cmd.end_rendering();
	}
	cmd.barrier( renderer::Stage::DEPTH, renderer::Stage::FRAGMENT );
}
