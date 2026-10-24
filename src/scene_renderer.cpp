#include "scene_renderer.h"

#include "camera.h"

#include <numeric>
#include <renderer/command_buffer.h>
#include <renderer/details/profiler.h>
#include <renderer/device.h>

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
	struct gpu_draw_mesh_header
	{
		uint32_t count;
	};
	struct gpu_draw_mesh_commands
	{
		std::array<gpu_draw_mesh_header, SceneRenderer::DRAW_BUCKETS_COUNT> headers;
		// Commands are stored as one array, each bucket starting at the sum of the previous buckets' capacities
		std::array<gpu_draw_mesh_command, MAX_DRAWS> commands;
	};

	// Mesh shader path, one per task workgroup
	struct gpu_task_command
	{
		uint32_t draw_id;
		uint32_t lod;
		uint32_t drawn_early;
		uint32_t first_meshlet;
	};
	struct gpu_task_header
	{
		VkDrawMeshTasksIndirectCommandEXT dispatch;
		uint32_t draw_count; // Always 1 for now
	};
	static_assert( offsetof( gpu_task_header, draw_count ) == offsetof( gpu_task_header, dispatch.groupCountZ ) + sizeof( uint32_t ),
				   "group_count_y, group_count_z and draw_count are reset to 1 by a single fill" );

	struct gpu_task_commands
	{
		std::array<gpu_task_header, SceneRenderer::DRAW_BUCKETS_COUNT> headers;
		std::array<gpu_task_command, SceneRenderer::MAX_TASK_COMMANDS> commands;
	};
	union gpu_draw_commands
	{
		gpu_draw_mesh_commands meshes;
		gpu_task_commands tasks;
	};

	template <typename Commands>
	constexpr size_t calc_header_offset( size_t bucket )
	{
		return offsetof( Commands, headers ) + bucket * sizeof( typename decltype( Commands::headers )::value_type );
	}

	struct draw_meshes_constants
	{
		renderer::BufferHandle scene_constants;
		renderer::BufferHandle draws_buffer;
		renderer::BufferHandle commands_buffer;
		// Task shader only
		renderer::BufferHandle meshlet_visibility;
		renderer::BindlessSampledTexture depth_pyramid;
		uint32_t meshlet_pass = 0;
	};

	struct grid_constants
	{
		glm::aligned_vec3 grid_color = { 0.1f, 0.1f, 0.1f };
		renderer::BufferHandle scene_constants;
		float near_distance = 50.f;
		float far_distance = 60.f;
	};

	struct depth_pyramid_constants
	{
		renderer::BindlessSampledTexture depth_in;
		renderer::BindlessStorageTexture depth_out;
		glm::aligned_vec2 size;
	};

	struct compute_draws_constants
	{
		renderer::BufferHandle draws_buffer;
		std::array<renderer::BufferHandle, SceneRenderer::DRAW_BUCKETS_COUNT> commands_buffers;
		std::array<renderer::BufferHandle, SceneRenderer::DRAW_BUCKETS_COUNT> commands_count_buffers;
		renderer::BufferHandle visibility_buffer;
		renderer::BufferHandle statistics_buffer;
		renderer::BufferHandle scene_constants;
		uint32_t draws_count;
		float lod_target;
		renderer::BindlessSampledTexture depth_pyramid;
		uint32_t meshlet_occlusion;
	};

	glm::vec4 normalize_plane( const glm::vec4& plane )
	{
		return plane / glm::length( glm::vec3( plane ) );
	}

	struct scene_constants
	{
		glm::mat4 view_projection_matrix;
		glm::mat4 projection_matrix;
		glm::mat4 view_matrix;
		glm::mat4 inverse_view_matrix;
		std::array<glm::mat4, SceneRenderer::SHADOW_CASCADE_MAX> shadow_view_projection_matrices;
		glm::aligned_vec4 shadow_cascade_splits; // Far view depth of each cascade
		glm::aligned_vec4 shadow_cascade_texel_sizes; // World size of a shadow map texel in each cascade
		std::array<glm::aligned_vec4, 6> frustum;
		glm::vec3 camera_position;
		uint32_t pbr;
		glm::aligned_vec4 main_light;
		float sky_to_sun;
		renderer::BindlessSampledTexture shadow_map;
		uint32_t shadow_pass = 0; // Non-zero when rendering a shadow map: use shadow (depth-only) LODs
		glm::aligned_vec2 viewport_size; // In pixels
		float shadow_distance = 0.f; // View depth up to which the main light casts shadows
		uint32_t shadow_cascade_count = 0;
		uint32_t show_shadow_cascades = 0;
	};
	static_assert( SceneRenderer::SHADOW_CASCADE_MAX == 4, "shadow_cascade_splits holds one split per vec4 component" );

	scene_constants build_scene_constants( const Camera& camera, renderer::Extent2D extent, glm::vec4 main_light, float sky_to_sun )
	{
		const glm::mat4 view = camera.calc_view_matrix();
		// https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
		const auto f = 1.0f / tan( SceneRenderer::FOV / 2.0f );
		const auto aspect_ratio = static_cast<float>( extent.width ) / extent.height;
		glm::mat4 projection( f / aspect_ratio, 0.0f, 0.0f, 0.0f, 0.0f, f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.01f, 0.0f );

		scene_constants constants { .view_projection_matrix = projection * view,
									.projection_matrix = projection,
									.view_matrix = view,
									.inverse_view_matrix = glm::inverse( view ),
									.camera_position = camera.position,
									.main_light = main_light,
									.sky_to_sun = sky_to_sun,
									.viewport_size = { extent.width, extent.height } };

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

	// Practical split scheme: blends uniform and logarithmic splits, lambda = 1 is fully logarithmic. Unused splits are far_depth.
	glm::vec4 compute_shadow_cascade_splits( uint32_t count, float near_depth, float far_depth, float lambda = 0.75f )
	{
		glm::vec4 splits( far_depth );
		for ( size_t i = 1; i < count; ++i )
		{
			const float t = static_cast<float>( i ) / static_cast<float>( count );
			const float uniform_split = near_depth + ( far_depth - near_depth ) * t;
			const float log_split = near_depth * std::pow( far_depth / near_depth, t );
			splits[ static_cast<glm::length_t>( i - 1 ) ] = glm::mix( uniform_split, log_split, lambda );
		}
		return splits;
	}

	// Fits a shadow map to the [near_depth, far_depth] slice of the camera frustum
	scene_constants build_scene_shadow_constants( const Camera& camera,
												  uint32_t resolution,
												  glm::vec3 light_direction,
												  float near_depth,
												  float far_depth,
												  renderer::Extent2D scene_extent )
	{
		const glm::mat4 inverse_camera_view = glm::inverse( camera.calc_view_matrix() );
		const auto aspect_ratio = static_cast<float>( scene_extent.width ) / scene_extent.height;
		const float tanY = tanf( SceneRenderer::FOV * 0.5f );
		const float tanX = tanY * aspect_ratio;
		std::vector<glm::vec3> bounds;
		bounds.reserve( 8 );
		for ( float d : { near_depth, far_depth } )
			for ( float y : { -1.f, 1.f } )
				for ( float x : { -1.f, 1.f } )
					bounds.emplace_back( inverse_camera_view * glm::vec4( x * tanX * d, y * tanY * d, -d, 1.f ) ); // camera looks down -Z

		auto center = std::accumulate( begin( bounds ), end( bounds ), glm::vec3 { } ) / static_cast<float>( bounds.size() );
		float radius = 0.f;
		for ( auto& p : bounds )
		{
			radius = glm::max( radius, glm::length( p - center ) );
		}
		radius = std::ceil( radius * 16.f ) / 16.f; // quantize so it doesn't jitter

		// Texel snapping magic (feat. Claude)
		const glm::vec3 up = fabsf( light_direction.y ) > 0.99f ? glm::vec3( 1, 0, 0 ) : glm::vec3( 0, 1, 0 );
		const glm::mat4 lightRot = glm::lookAtRH( glm::vec3( 0 ), light_direction, up ); // rotation only
		const float texel = 2.f * radius / static_cast<float>( resolution );
		glm::vec3 cLS = glm::vec3( lightRot * glm::vec4( center, 1.f ) );
		cLS.x = std::floor( cLS.x / texel ) * texel;
		cLS.y = std::floor( cLS.y / texel ) * texel;
		center = glm::vec3( glm::inverse( lightRot ) * glm::vec4( cLS, 1.f ) );

		float backOff = radius + 10.f;
		const glm::vec3 eye = center - light_direction * backOff;
		const glm::mat4 view = glm::lookAtRH( eye, center, up );
		const glm::mat4 projection = glm::orthoRH_ZO( -radius, radius, -radius, radius, 0.f, backOff + radius );

		scene_constants constants { .view_projection_matrix = projection * view,
									.projection_matrix = projection,
									.view_matrix = view,
									.inverse_view_matrix = glm::inverse( view ),
									.camera_position = camera.position,
									.main_light = { 0.f, 0.f, 0.f, -1.f },
									.shadow_pass = 1,
									.viewport_size = { resolution, resolution } };

		const auto transposed_vp = glm::transpose( constants.view_projection_matrix );
		constants.frustum[ 0 ] = normalize_plane( transposed_vp[ 3 ] + transposed_vp[ 0 ] ); // x + w < 0
		constants.frustum[ 1 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 0 ] ); // x - w > 0
		constants.frustum[ 2 ] = normalize_plane( transposed_vp[ 3 ] + transposed_vp[ 1 ] ); // y + w < 0
		constants.frustum[ 3 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 1 ] ); // y - w > 0
		constants.frustum[ 4 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 2 ] ); // far    (w - z >= 0, depth <= 1)
		constants.frustum[ 5 ] = glm::vec4( 0, 0, 0, 1 ); // always passes; near plane intentionally skipped

		return constants;
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
							  BindlessManager& bindless_manager,
							  renderer::PipelineManager& pipelines )
	: _extent( device.get_extent() )
	, _depth_pyramid_extent( round_pow2( _extent.width ), round_pow2( _extent.height ) )
	, _color_texture( device.create_texture( renderer::Texture::Desc { .format = COLOR_FORMAT,
																	   .usage = usage::COLOR_ATTACHMENT,
																	   .extent = _extent,
																	   .samples = SAMPLE_COUNT } ) )
	, _color_texture_view( device.create_texture_view( _color_texture, renderer::TextureView::Aspect::COLOR ) )
	, _depth_texture( device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::D32_SFLOAT,
																	   .usage = usage::DEPTH_STENCIL_ATTACHMENT,
																	   .extent = _extent,
																	   .samples = SAMPLE_COUNT } ) )
	, _depth_texture_view( device.create_texture_view( _depth_texture, renderer::TextureView::Aspect::DEPTH ) )
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
	, _constant_buffers( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
											   sizeof( std::array<scene_constants, renderer::MAX_FRAMES_IN_FLIGHT> ),
											   renderer::Buffer::CpuUsage::WRITE ) )
	, _shadowmap_constant_buffers(
		  device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
								sizeof( std::array<scene_constants, renderer::MAX_FRAMES_IN_FLIGHT * SHADOW_CASCADE_MAX> ),
								renderer::Buffer::CpuUsage::WRITE ) )
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
																			 { "USE_ALPHA", "false" } };
	const std::vector<renderer::ShaderSource::Define> mesh_alpha_shaders_defines { { "MESH_LOD_COUNT", std::to_string( MESH_LOD_COUNT ) },
																				   { "TASK_GROUP_SIZE", std::to_string( TASK_GROUP_SIZE ) },
																				   { "USE_ALPHA", "true" } };

	_mesh_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
								   .depth_format = DEPTH_FORMAT,
								   .sample_count = SAMPLE_COUNT,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines },
		  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

	_mesh_alpha_clip_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
								   .depth_format = DEPTH_FORMAT,
								   .sample_count = SAMPLE_COUNT,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines },
		  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_alpha_shaders_defines } } );

	_mesh_alpha_blend_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
								   .depth_format = DEPTH_FORMAT,
								   .depth_compare_op = renderer::CompareOp::GREATER_OR_EQUAL,
								   .sample_count = SAMPLE_COUNT,
								   .alpha_blend = true,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines },
		  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

	_shadowmap_mesh_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .depth_format = SHADOW_DEPTH_FORMAT,
								   .depth_compare_op = renderer::CompareOp::LESS,
								   .depth_clamp = true,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines } } );

	if ( device.get_properties().mesh_shader_support )
	{
		_meshlet_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
									   .depth_format = DEPTH_FORMAT,
									   .sample_count = SAMPLE_COUNT,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines },
			  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

		_meshlet_alpha_clip_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
									   .depth_format = DEPTH_FORMAT,
									   .sample_count = SAMPLE_COUNT,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines },
			  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_alpha_shaders_defines } } );

		_meshlet_alpha_blend_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
									   .depth_format = DEPTH_FORMAT,
									   .depth_compare_op = renderer::CompareOp::GREATER_OR_EQUAL,
									   .sample_count = SAMPLE_COUNT,
									   .alpha_blend = true,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines },
			  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

		_shadowmap_meshlet_pipeline = pipelines.add(
			renderer::Pipeline::Desc { .depth_format = SHADOW_DEPTH_FORMAT,
									   .depth_compare_op = renderer::CompareOp::LESS,
									   .depth_clamp = true,
									   .push_constants_size = sizeof( draw_meshes_constants ) },
			{ { .path = "mesh.task", .stage = renderer::ShaderStage::TASK, .defines = mesh_shaders_defines },
			  { .path = "mesh.mesh", .stage = renderer::ShaderStage::MESH, .defines = mesh_shaders_defines } } );
	}

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

	_depth_pyramid_pipeline = pipelines.add( renderer::Pipeline::Desc { .push_constants_size = sizeof( depth_pyramid_constants ) },
											 { { .path = "depth_pyramid.comp", .stage = renderer::ShaderStage::COMPUTE } } );

	_grid_pipeline = pipelines.add( renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
															   .depth_format = DEPTH_FORMAT,
															   .topology = renderer::Pipeline::PrimitiveTopology::TRIANGLE_STRIP,
															   .sample_count = SAMPLE_COUNT,
															   .push_constants_size = sizeof( grid_constants ) },
									{ { .path = "grid.vert", .stage = renderer::ShaderStage::VERTEX },
									  { .path = "grid.frag", .stage = renderer::ShaderStage::FRAGMENT } } );

	_tonemap_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = output_format, .push_constants_size = sizeof( renderer::BindlessSampledTexture ) },
		{ { .path = "fullscreen.vert", .stage = renderer::ShaderStage::VERTEX },
		  { .path = "tonemap.frag", .stage = renderer::ShaderStage::FRAGMENT } } );

	for ( const auto buffer : { &_draw_commands_buffer, &_shadowmap_draw_commands_buffer } )
	{
		*buffer = device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::INDIRECT_BUFFER
											| renderer::Buffer::Usage::TRANSFER_DST | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
										sizeof( gpu_draw_commands ) );
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

void SceneRenderer::draw_bucket( renderer::CommandBuffer& cmd,
								 const renderer::Buffer& draw_commands,
								 size_t bucket,
								 bool use_mesh_shader ) const
{
	const auto& segment = get_draw_commands_segments( use_mesh_shader )[ bucket ];
	if ( segment.capacity == 0 )
	{
		return;
	}

	if ( use_mesh_shader )
	{
		// All the task workgroups in a single dispatch, a group count of 0 draws nothing
		const auto header_offset = calc_header_offset<gpu_task_commands>( bucket );
		cmd.draw_mesh_tasks_indirect( draw_commands,
									  header_offset + offsetof( gpu_task_header, dispatch ),
									  draw_commands,
									  header_offset + offsetof( gpu_task_header, draw_count ),
									  1,
									  sizeof( VkDrawMeshTasksIndirectCommandEXT ) );
	}
	else if ( _draw_indirect_count )
	{
		cmd.draw_indexed_indirect( draw_commands,
								   segment.commands_offset + offsetof( gpu_draw_mesh_command, draw_cmd ),
								   draw_commands,
								   calc_header_offset<gpu_draw_mesh_commands>( bucket ) + offsetof( gpu_draw_mesh_header, count ),
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
							const Camera& camera,
							glm::vec4 main_light,
							float sky_illumination,
							float lod_pixel_tolerance,
							float shadow_distance,
							uint32_t shadow_cascade_count,
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

	if ( use_occlusion_culling )
	{
		if ( _needs_rebuild )
		{
			// Can't reuse visibility buffer if the scene has changed
			cmd.fill_buffer( _draw_visibility_buffer, 0, _draw_visibility_buffer.get_size(), 0 );
			cmd.buffer_barrier( _draw_visibility_buffer, renderer::ResourceState::TRANSFER_DST, renderer::ResourceState::STORAGE_READ );
			_needs_rebuild = false;
		}
	}
	else
	{
		// Force everything to be drawn in the early pass
		cmd.fill_buffer( _draw_visibility_buffer, 0, _draw_visibility_buffer.get_size(), 0x01010101 );
		cmd.buffer_barrier( _draw_visibility_buffer, renderer::ResourceState::TRANSFER_DST, renderer::ResourceState::STORAGE_READ );
		_needs_rebuild = true;
	}

	// Previous frame copied it to its readback buffer
	cmd.buffer_barrier( _draw_statistics_buffer, renderer::ResourceState::TRANSFER_SRC, renderer::ResourceState::TRANSFER_DST );
	cmd.fill_buffer( _draw_statistics_buffer, 0, sizeof( DrawStatistics ), 0 );
	cmd.buffer_barrier( _draw_statistics_buffer, renderer::ResourceState::TRANSFER_DST, renderer::ResourceState::STORAGE_READ_WRITE );

	using Draws = std::array<MeshDraw, MAX_DRAWS>;
	memcpy( static_cast<Draws*>( _draw_buffers.get_mapped_address() ) + frame_index, _scene.data(), sizeof( MeshDraw ) * _scene.size() );
	const auto draw_buffer_address = _draw_buffers.get_device_address() + ( frame_index * sizeof( Draws ) );

	ShadowCascades shadow_cascades;
	if ( shadow_cascade_count > 0 )
	{
		const auto label = cmd.debug_label( "Shadow Map" );
		shadow_cascades = draw_shadowmap( bindless_manager,
										  pipelines,
										  cmd,
										  draw_buffer_address,
										  _draw_statistics_buffer.get_device_address(),
										  camera,
										  main_light,
										  shadow_distance,
										  std::min( shadow_cascade_count, SHADOW_CASCADE_MAX ),
										  index_buffer == nullptr );
	}

	{
		auto constants = build_scene_constants( camera, _extent, main_light, sky_illumination );
		if ( shadow_cascades.count > 0 )
		{
			constants.shadow_view_projection_matrices = shadow_cascades.view_projections;
			constants.shadow_cascade_splits = shadow_cascades.splits;
			constants.shadow_cascade_texel_sizes = shadow_cascades.texel_sizes;
			constants.shadow_cascade_count = shadow_cascades.count;
			constants.show_shadow_cascades = show_shadow_cascades ? 1 : 0;
			constants.shadow_map = _shadowmap.handles.sampled_texture;
			constants.shadow_distance = shadow_distance;
		}
		constants.pbr = use_pbr ? 1 : 0;
		memcpy( static_cast<scene_constants*>( _constant_buffers.get_mapped_address() ) + frame_index, &constants, sizeof( constants ) );
	}
	const auto constants_address = _constant_buffers.get_device_address() + ( frame_index * sizeof( scene_constants ) );

	const bool use_mesh_shader = index_buffer == nullptr;
	const bool meshlet_occlusion = use_occlusion_culling && use_mesh_shader;
	if ( meshlet_occlusion )
	{
		// Previous frame's late pass wrote the bits the early pass reads
		cmd.buffer_barrier( _meshlet_visibility_buffer,
							renderer::ResourceState::SHADER_STORAGE_READ_WRITE,
							renderer::ResourceState::SHADER_STORAGE_READ_WRITE );
	}

	{
		const auto label = cmd.debug_label( "Opaque Early" );
		cmd.texture_barrier( _color_texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::COLOR_ATTACHMENT );
		cmd.texture_barrier( _depth_texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::DEPTH_ATTACHMENT );
		cmd.texture_barrier( _resolved_depth_texture.texture,
							 renderer::ResourceState::UNDEFINED,
							 renderer::ResourceState::DEPTH_ATTACHMENT );

		//cmd.write_timestamp( timestamp_queries[ image_index ], early ? Timestamps::COMPUTE_DRAWS_EARLY : Timestamps::COMPUTE_DRAWS_LATE );
		compute_draws( bindless_manager,
					   pipelines,
					   cmd,
					   constants_address,
					   draw_buffer_address,
					   _draw_visibility_buffer,
					   _draw_statistics_buffer.get_device_address(),
					   _draw_commands_buffer,
					   lod_pixel_tolerance / ( _extent.height / ( 2 * std::tan( FOV * 0.5f ) ) ),
					   true,
					   meshlet_occlusion,
					   use_mesh_shader );

		cmd.begin_rendering( { .color_target = { .target = _color_texture_view, .clear_value = { { 0.1f, 0.1f, 0.1f, 1.f } } },
							   .depth_target = { .target = _depth_texture_view,
												 .resolve_target = _resolved_depth_texture.handles.view,
												 .clear_value = { { 0.f } } } } );
		draw_opaques( bindless_manager,
					  pipelines,
					  cmd,
					  constants_address,
					  draw_buffer_address,
					  use_mesh_shader,
					  meshlet_occlusion ? MeshletPass::EARLY : MeshletPass::NONE );
		cmd.end_rendering();
	}


	if ( use_occlusion_culling )
	{
		cmd.texture_barrier( _resolved_depth_texture.texture,
							 renderer::ResourceState::DEPTH_ATTACHMENT,
							 renderer::ResourceState::COMPUTE_SHADER_READ );

		{
			//cmd.write_timestamp( timestamp_queries[ image_index ], Timestamps::DEPTH_PYRAMID );
			PROFILER_GPU_EVENT( "Depth Pyramid" );
			const auto label = cmd.debug_label( "Depth Pyramid" );
			cmd.bind_pipeline( pipelines.get( _depth_pyramid_pipeline ), bindless_manager );
			auto extent = _depth_pyramid_extent;
			for ( size_t mip = 0; mip < _depth_pyramid.mips.size(); ++mip )
			{
				cmd.texture_barrier( _depth_pyramid.texture,
									 renderer::ResourceState::UNDEFINED,
									 renderer::ResourceState::STORAGE_WRITE,
									 static_cast<int>( mip ) );

				const depth_pyramid_constants constants { .depth_in = mip == 0 ? _resolved_depth_texture.handles.sampled_texture
																			   : _depth_pyramid.mips[ mip - 1 ].sampled_texture,
														  .depth_out = _depth_pyramid.mips[ mip ].storage_texture,
														  .size = glm::aligned_vec2( extent.width, extent.height ) };
				cmd.push_constants( pipelines.get( _depth_pyramid_pipeline ), constants );
				cmd.dispatch( calc_group_count( extent.width ), calc_group_count( extent.height ), 1 );

				cmd.texture_barrier( _depth_pyramid.texture,
									 renderer::ResourceState::STORAGE_WRITE,
									 renderer::ResourceState::COMPUTE_SHADER_READ,
									 static_cast<int>( mip ) );

				extent.width = std::max( extent.width / 2, 1u );
				extent.height = std::max( extent.height / 2, 1u );
			}
		}

		const auto label = cmd.debug_label( "Opaque Late" );
		compute_draws( bindless_manager,
					   pipelines,
					   cmd,
					   constants_address,
					   draw_buffer_address,
					   _draw_visibility_buffer,
					   _draw_statistics_buffer.get_device_address(),
					   _draw_commands_buffer,
					   lod_pixel_tolerance / ( _extent.height / ( 2 * std::tan( FOV * 0.5f ) ) ),
					   false,
					   meshlet_occlusion,
					   use_mesh_shader );

		if ( meshlet_occlusion )
		{
			cmd.buffer_barrier( _meshlet_visibility_buffer,
								renderer::ResourceState::SHADER_STORAGE_READ_WRITE,
								renderer::ResourceState::SHADER_STORAGE_READ_WRITE );
		}
		cmd.begin_rendering( { .color_target = { .target = _color_texture_view }, .depth_target = { .target = _depth_texture_view } } );
		draw_opaques( bindless_manager,
					  pipelines,
					  cmd,
					  constants_address,
					  draw_buffer_address,
					  use_mesh_shader,
					  meshlet_occlusion ? MeshletPass::LATE : MeshletPass::NONE );
		cmd.end_rendering();
	}

	{
		const auto label = cmd.debug_label( "Transparents" );
		cmd.begin_rendering( { .color_target = { .target = _color_texture_view }, .depth_target = { .target = _depth_texture_view } } );
		if ( !index_buffer )
		{
			draw_transparent_meshlets( bindless_manager, pipelines, cmd, constants_address, draw_buffer_address );
		}
		else
		{
			draw_transparent_meshes( bindless_manager, pipelines, cmd, constants_address, draw_buffer_address );
		}
		cmd.end_rendering();
	}

	cmd.buffer_barrier( _draw_statistics_buffer, renderer::ResourceState::STORAGE_READ_WRITE, renderer::ResourceState::TRANSFER_SRC );
	cmd.copy_buffer( _draw_statistics_buffer, 0, sizeof( DrawStatistics ), _draw_statistics_readback_buffers[ frame_index ] );
	cmd.buffer_barrier( _draw_statistics_readback_buffers[ frame_index ],
						renderer::ResourceState::TRANSFER_DST,
						renderer::ResourceState::HOST_READ );

	{
		PROFILER_GPU_EVENT( "Draw Grid" );
		const auto label = cmd.debug_label( "Grid" );
		cmd.texture_barrier( _resolved_color_texture.texture,
							 renderer::ResourceState::UNDEFINED,
							 renderer::ResourceState::COLOR_ATTACHMENT );
		cmd.begin_rendering( { .color_target = { .target = _color_texture_view, .resolve_target = _resolved_color_texture.handles.view },
							   .depth_target = { .target = _depth_texture_view } } );

		cmd.set_dynamic_state( { .viewport = _extent } );
		//cmd.write_timestamp( timestamp_queries[ image_index ], Timestamps::DRAW_GRID );
		cmd.bind_pipeline( pipelines.get( _grid_pipeline ), bindless_manager );
		cmd.push_constants( pipelines.get( _grid_pipeline ), grid_constants { .scene_constants = constants_address } );
		cmd.draw( 4 );
		cmd.end_rendering();
	}

	{
		PROFILER_GPU_EVENT( "Tonemap" );
		const auto label = cmd.debug_label( "Tonemap" );
		//cmd.write_timestamp( timestamp_queries[ image_index ], Timestamps::BLIT );
		cmd.texture_barrier( _resolved_color_texture.texture,
							 renderer::ResourceState::COLOR_ATTACHMENT,
							 renderer::ResourceState::FRAGMENT_SHADER_READ );
		cmd.texture_barrier( output_texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::COLOR_ATTACHMENT );
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
								   renderer::BufferHandle constants_address,
								   renderer::BufferHandle draw_buffer_address,
								   const renderer::Buffer& visibility_buffer,
								   renderer::BufferHandle stats_buffer_address,
								   const renderer::Buffer& draw_commands,
								   float lod_target,
								   bool early,
								   bool meshlet_occlusion,
								   bool task_commands ) const
{
	PROFILER_GPU_EVENT( "Compute Draws" );
	const auto label = cmd.debug_label( "Culling" );
	compute_draws_constants constants { .draws_buffer = draw_buffer_address,
										.visibility_buffer = visibility_buffer.get_device_address(),
										.statistics_buffer = stats_buffer_address,
										.scene_constants = constants_address,
										.draws_count = static_cast<uint32_t>( _scene.size() ),
										.lod_target = lod_target,
										.meshlet_occlusion = meshlet_occlusion ? 1u : 0u };

	if ( !early )
	{
		constants.depth_pyramid = _depth_pyramid.handles.sampled_texture;
	}

	// Previous draws read commands both as indirect arguments and from shaders
	cmd.buffer_barrier( draw_commands, renderer::ResourceState::ANY_BUFFER_READ, renderer::ResourceState::TRANSFER_DST );
	if ( !task_commands )
	{
		// Draws without draw indirect count support read all the buckets' commands, commands past the count must be empty
		const size_t commands_size = _draw_indirect_count ? 0 : _scene.size() * sizeof( gpu_draw_mesh_command );
		cmd.fill_buffer( draw_commands, 0, offsetof( gpu_draw_mesh_commands, commands ) + commands_size, 0 );
	}
	for ( size_t i = 0; i < DRAW_BUCKETS_COUNT; ++i )
	{
		const auto header_offset = task_commands ? calc_header_offset<gpu_task_commands>( i )
												 : calc_header_offset<gpu_draw_mesh_commands>( i );
		if ( task_commands )
		{
			// Group count 0, Y, Z and draw count 1. Commands past the group count are never read
			cmd.fill_buffer( draw_commands, header_offset + offsetof( gpu_task_header, dispatch.groupCountX ), sizeof( uint32_t ), 0 );
			cmd.fill_buffer( draw_commands, header_offset + offsetof( gpu_task_header, dispatch.groupCountY ), 3 * sizeof( uint32_t ), 1 );
		}
		constants.commands_buffers[ i ] = draw_commands.get_device_address()
			+ get_draw_commands_segments( task_commands )[ i ].commands_offset;
		constants.commands_count_buffers[ i ] = draw_commands.get_device_address() + header_offset;
	}
	cmd.buffer_barrier( draw_commands, renderer::ResourceState::TRANSFER_DST, renderer::ResourceState::STORAGE_READ_WRITE );

	const auto pipeline_handle = task_commands ? ( early ? _compute_tasks_early_pipeline : _compute_tasks_late_pipeline )
											   : ( early ? _compute_draws_early_pipeline : _compute_draws_late_pipeline );
	const auto pipeline = pipelines.get( pipeline_handle );
	cmd.bind_pipeline( pipeline, bindless_manager );

	cmd.push_constants( pipeline, constants );
	cmd.dispatch( calc_group_count( static_cast<uint32_t>( _scene.size() ) ), 1, 1 );
	// Task and vertex shaders also read the commands
	cmd.buffer_barrier( draw_commands, renderer::ResourceState::STORAGE_WRITE, renderer::ResourceState::ANY_BUFFER_READ );
	cmd.buffer_barrier( visibility_buffer, renderer::ResourceState::STORAGE_READ_WRITE, renderer::ResourceState::STORAGE_READ_WRITE );
}

void SceneRenderer::draw_opaques( const BindlessManager& bindless_manager,
								  const renderer::PipelineManager& pipelines,
								  renderer::CommandBuffer& cmd,
								  renderer::BufferHandle constants_address,
								  renderer::BufferHandle draw_buffer_address,
								  bool use_mesh_shader,
								  MeshletPass meshlet_pass ) const
{
	const auto label = cmd.debug_label( "Draw" );

	//cmd.write_timestamp( timestamp_queries[ image_index ], early ? Timestamps::DRAW_MESHES_EARLY : Timestamps::DRAW_MESHES_LATE );
	if ( use_mesh_shader )
	{
		draw_opaque_meshlets( bindless_manager, pipelines, cmd, constants_address, draw_buffer_address, meshlet_pass );
	}
	else
	{
		draw_opaque_meshes( bindless_manager, pipelines, cmd, constants_address, draw_buffer_address );
	}
}

void SceneRenderer::draw_opaque_meshlets( const BindlessManager& bindless_manager,
										  const renderer::PipelineManager& pipelines,
										  renderer::CommandBuffer& cmd,
										  renderer::BufferHandle constants_address,
										  renderer::BufferHandle draw_buffer_address,
										  MeshletPass meshlet_pass ) const
{
	PROFILER_GPU_EVENT( "Draw Meshlets" );
	cmd.bind_pipeline( pipelines.get( _meshlet_pipeline ), bindless_manager );
	draw_meshes_constants constants { .scene_constants = constants_address,
									  .draws_buffer = draw_buffer_address,
									  .meshlet_visibility = _meshlet_visibility_buffer.get_device_address(),
									  .depth_pyramid = _depth_pyramid.handles.sampled_texture,
									  .meshlet_pass = std::to_underlying( meshlet_pass ) };
	for ( int i = 0; i < 2; ++i )
	{
		const auto bucket = std::to_underlying( material_type::OPAQUE_SINGLE );
		cmd.set_dynamic_state( { .viewport = _extent, .cull_mode = bucket_cull_mode( i ) } );
		constants.commands_buffer = _draw_commands_buffer.get_device_address() + _task_segments[ bucket + i ].commands_offset;
		cmd.push_constants( pipelines.get( _meshlet_pipeline ), constants );
		draw_bucket( cmd, _draw_commands_buffer, bucket + i, true );
	}
}

void SceneRenderer::draw_opaque_meshes( const BindlessManager& bindless_manager,
										const renderer::PipelineManager& pipelines,
										renderer::CommandBuffer& cmd,
										renderer::BufferHandle constants_address,
										renderer::BufferHandle draw_buffer_address ) const
{
	PROFILER_GPU_EVENT( "Draw Meshes" );
	cmd.bind_pipeline( pipelines.get( _mesh_pipeline ), bindless_manager );
	draw_meshes_constants constants { .scene_constants = constants_address, .draws_buffer = draw_buffer_address };
	for ( int i = 0; i < 2; ++i )
	{
		const auto bucket = std::to_underlying( material_type::OPAQUE_SINGLE );
		cmd.set_dynamic_state( { .viewport = _extent, .cull_mode = bucket_cull_mode( i ) } );
		constants.commands_buffer = _draw_commands_buffer.get_device_address() + _draw_mesh_segments[ bucket + i ].commands_offset;
		cmd.push_constants( pipelines.get( _mesh_pipeline ), constants );
		draw_bucket( cmd, _draw_commands_buffer, bucket + i, false );
	}
}

void SceneRenderer::draw_transparent_meshlets( const BindlessManager& bindless_manager,
											   const renderer::PipelineManager& pipelines,
											   renderer::CommandBuffer& cmd,
											   renderer::BufferHandle constants_address,
											   renderer::BufferHandle draw_buffer_address ) const
{
	PROFILER_GPU_EVENT( "Draw Transparent Meshlets" );
	draw_meshes_constants constants { .scene_constants = constants_address, .draws_buffer = draw_buffer_address };
	for ( int pass = 0; pass < 2; ++pass )
	{
		const auto pipeline = pipelines.get( pass == 0 ? _meshlet_alpha_clip_pipeline : _meshlet_alpha_blend_pipeline );
		cmd.bind_pipeline( pipeline, bindless_manager );
		for ( int i = 0; i < 2; ++i )
		{
			const auto bucket = std::to_underlying( material_type::ALPHA_SINGLE );
			cmd.set_dynamic_state( { .viewport = _extent, .cull_mode = bucket_cull_mode( i ), .depth_write = pass == 0 } );
			constants.commands_buffer = _draw_commands_buffer.get_device_address() + _task_segments[ bucket + i ].commands_offset;
			cmd.push_constants( pipeline, constants );
			draw_bucket( cmd, _draw_commands_buffer, bucket + i, true );
		}
	}
}

void SceneRenderer::draw_transparent_meshes( const BindlessManager& bindless_manager,
											 const renderer::PipelineManager& pipelines,
											 renderer::CommandBuffer& cmd,
											 renderer::BufferHandle constants_address,
											 renderer::BufferHandle draw_buffer_address ) const
{
	PROFILER_GPU_EVENT( "Draw Transparent Meshes" );
	draw_meshes_constants constants { .scene_constants = constants_address, .draws_buffer = draw_buffer_address };
	for ( int pass = 0; pass < 2; ++pass )
	{
		const auto pipeline = pipelines.get( pass == 0 ? _mesh_alpha_clip_pipeline : _mesh_alpha_blend_pipeline );
		cmd.bind_pipeline( pipeline, bindless_manager );

		for ( int i = 0; i < 2; ++i )
		{
			const auto bucket = std::to_underlying( material_type::ALPHA_SINGLE );
			cmd.set_dynamic_state( { .viewport = _extent, .cull_mode = bucket_cull_mode( i ), .depth_write = pass == 0 } );
			constants.commands_buffer = _draw_commands_buffer.get_device_address() + _draw_mesh_segments[ bucket + i ].commands_offset;
			cmd.push_constants( pipeline, constants );
			draw_bucket( cmd, _draw_commands_buffer, bucket + i, false );
		}
	}
}

SceneRenderer::ShadowCascades SceneRenderer::draw_shadowmap( const BindlessManager& bindless_manager,
															 const renderer::PipelineManager& pipelines,
															 renderer::CommandBuffer& cmd,
															 renderer::BufferHandle draw_buffer_address,
															 renderer::BufferHandle stats_buffer_address,
															 const Camera& camera,
															 glm::vec3 main_light,
															 float shadow_distance,
															 uint32_t cascade_count,
															 bool use_mesh_shader )
{
	const auto frame_index = _frames % renderer::MAX_FRAMES_IN_FLIGHT;
	const auto shadowmap_resolution = _shadowmap.texture.get_extent().width;
	constexpr float shadow_near_depth = 0.1f;
	ShadowCascades cascades { .splits = compute_shadow_cascade_splits( cascade_count, shadow_near_depth, shadow_distance ),
							  .count = cascade_count };

	if ( _frames == 0 )
	{
		cmd.fill_buffer( _shadowmap_visibility_buffer, 0, _shadowmap_visibility_buffer.get_size(), 0x01010101 );
		cmd.buffer_barrier( _shadowmap_visibility_buffer, renderer::ResourceState::TRANSFER_DST, renderer::ResourceState::STORAGE_READ );
	}

	cmd.texture_barrier( _shadowmap.texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::DEPTH_ATTACHMENT );
	const auto& pipeline = pipelines.get( use_mesh_shader ? _shadowmap_meshlet_pipeline : _shadowmap_mesh_pipeline );

	static constexpr std::array<const char*, SHADOW_CASCADE_MAX> cascade_labels { "Cascade 0", "Cascade 1", "Cascade 2", "Cascade 3" };
	for ( size_t cascade = 0; cascade < cascade_count; ++cascade )
	{
		const auto cascade_label = cmd.debug_label( cascade_labels[ cascade ] );
		const float near_depth = cascade == 0 ? shadow_near_depth : cascades.splits[ static_cast<glm::length_t>( cascade - 1 ) ];
		const auto constants = build_scene_shadow_constants( camera,
															 shadowmap_resolution,
															 main_light,
															 near_depth,
															 cascades.splits[ static_cast<glm::length_t>( cascade ) ],
															 _extent );
		cascades.view_projections[ cascade ] = constants.view_projection_matrix;
		const auto constants_index = frame_index * SHADOW_CASCADE_MAX + cascade;
		memcpy( static_cast<scene_constants*>( _shadowmap_constant_buffers.get_mapped_address() ) + constants_index,
				&constants,
				sizeof( constants ) );
		const auto constants_address = _shadowmap_constant_buffers.get_device_address() + ( constants_index * sizeof( scene_constants ) );

		const float texel_world_size = 2.f / ( constants.projection_matrix[ 0 ][ 0 ] * shadowmap_resolution );
		cascades.texel_sizes[ static_cast<glm::length_t>( cascade ) ] = texel_world_size;
		const float lod_target = texel_world_size * 2.f;

		compute_draws( bindless_manager,
					   pipelines,
					   cmd,
					   constants_address,
					   draw_buffer_address,
					   _shadowmap_visibility_buffer,
					   stats_buffer_address,
					   _shadowmap_draw_commands_buffer,
					   lod_target,
					   true,
					   false,
					   use_mesh_shader );

		PROFILER_GPU_EVENT( "Draw Shadows" );
		const auto label = cmd.debug_label( "Draw" );
		cmd.begin_rendering( { .depth_target = { .target = _shadowmap_layer_views[ cascade ], .clear_value = { { 1.f } } } } );
		cmd.bind_pipeline( pipeline, bindless_manager );
		draw_meshes_constants draw_constants { .scene_constants = constants_address, .draws_buffer = draw_buffer_address };
		for ( int i = 0; i < 2; ++i )
		{
			const auto bucket = std::to_underlying( material_type::OPAQUE_SINGLE );
			cmd.set_dynamic_state( { .viewport = _shadowmap.texture.get_extent(),
									 .cull_mode = bucket_cull_mode( i ),
									 .depth_bias_constant = SHADOW_DEPTH_BIAS_CONSTANT,
									 .depth_bias_slope = SHADOW_DEPTH_BIAS_SLOPE } );
			draw_constants.commands_buffer = _shadowmap_draw_commands_buffer.get_device_address()
				+ get_draw_commands_segments( use_mesh_shader )[ bucket + i ].commands_offset;
			cmd.push_constants( pipeline, draw_constants );
			draw_bucket( cmd, _shadowmap_draw_commands_buffer, bucket + i, use_mesh_shader );
		}
		cmd.end_rendering();
	}
	cmd.texture_barrier( _shadowmap.texture, renderer::ResourceState::DEPTH_ATTACHMENT, renderer::ResourceState::FRAGMENT_SHADER_READ );
	return cascades;
}
