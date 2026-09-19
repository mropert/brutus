#include "scene_renderer.h"

#include "camera.h"

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

	// Not used on CPU side, only there for buffer sizing
	struct gpu_draw_command
	{
		uint32_t draw_id;
		uint32_t lod;
		VkDrawIndexedIndirectCommand draw_cmd;
		VkDrawMeshTasksIndirectCommandEXT mesh_cmd;
	};
	static constexpr auto MAX_DRAWS = 1024 * 1024;
	struct DrawCommands
	{
		uint32_t count;
		std::array<gpu_draw_command, MAX_DRAWS> commands;
	};

	struct draw_meshes_constants
	{
		renderer::BufferHandle scene_constants;
		renderer::BufferHandle draws_buffer;
		renderer::BufferHandle commands_buffer;
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
		renderer::BufferHandle commands_buffer;
		renderer::BufferHandle commmands_count_buffer;
		renderer::BufferHandle visibility_buffer;
		renderer::BufferHandle statistics_buffer;
		renderer::BufferHandle scene_constants;
		uint32_t draws_count;
		float lod_target;
		renderer::BindlessSampledTexture depth_pyramid;
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
		std::array<glm::aligned_vec4, 6> frustum;
		glm::vec3 camera_position;
		uint32_t pbr;
		glm::aligned_vec4 main_light;
		float sky_to_sun;
	};

	scene_constants build_scene_constants( const Camera& camera, renderer::Extent2D extent, glm::vec4 main_light, float sky_to_sun )
	{
		const glm::mat4 view = camera.calc_view_matrix();
		// https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
		const auto f = 1.0f / tan( 70.f / 2.0f );
		const auto aspect_ratio = static_cast<float>( extent.width ) / extent.height;
		glm::mat4 projection( f / aspect_ratio, 0.0f, 0.0f, 0.0f, 0.0f, f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.01f, 0.0f );

		scene_constants constants { .view_projection_matrix = projection * view,
									.projection_matrix = projection,
									.view_matrix = view,
									.inverse_view_matrix = glm::inverse( view ),
									.camera_position = camera.position,
									.main_light = main_light,
									.sky_to_sun = sky_to_sun };

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
	, _constant_buffers( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
											   sizeof( std::array<scene_constants, renderer::MAX_FRAMES_IN_FLIGHT> ),
											   renderer::Buffer::CpuUsage::WRITE ) )
	, _draw_buffers( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
										   sizeof( std::array<MeshDraw, MAX_DRAWS> ) * renderer::MAX_FRAMES_IN_FLIGHT,
										   renderer::Buffer::CpuUsage::WRITE ) )
	, _draw_visibility_buffer( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
														 | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
													 MAX_DRAWS ) )
	, _draw_commands_buffer( device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::INDIRECT_BUFFER
													   | renderer::Buffer::Usage::TRANSFER_DST
													   | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
												   sizeof( DrawCommands ) ) )
	, _draw_indirect_count( device.get_properties().draw_indirect_count_support )
{
	const std::vector<renderer::ShaderSource::Define> mesh_shaders_defines { { "MESH_LOD_COUNT", std::to_string( MESH_LOD_COUNT ) } };

	_mesh_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = COLOR_FORMAT,
								   .depth_format = DEPTH_FORMAT,
								   .sample_count = SAMPLE_COUNT,
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { .path = "mesh.vert", .stage = renderer::ShaderStage::VERTEX, .defines = mesh_shaders_defines },
		  { .path = "mesh.frag", .stage = renderer::ShaderStage::FRAGMENT, .defines = mesh_shaders_defines } } );

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
	}

	std::vector<renderer::ShaderSource::Define> compute_shader_defines;
	compute_shader_defines.emplace_back( "LATE", "false" );
	compute_shader_defines.emplace_back( "EMIT_EMPTY", device.get_properties().draw_indirect_count_support ? "false" : "true" );
	std::copy( begin( mesh_shaders_defines ), end( mesh_shaders_defines ), std::back_inserter( compute_shader_defines ) );

	_compute_draws_early_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .push_constants_size = sizeof( compute_draws_constants ) },
		{ { .path = "mesh_cull.comp", .stage = renderer::ShaderStage::COMPUTE, .defines = compute_shader_defines } } );
	compute_shader_defines[ 0 ].value = "true";
	_compute_draws_late_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .push_constants_size = sizeof( compute_draws_constants ) },
		{ { .path = "mesh_cull.comp", .stage = renderer::ShaderStage::COMPUTE, .defines = compute_shader_defines } } );

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

	for ( auto& buffer : _draw_statistics_buffers )
	{
		buffer = device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
										   | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
									   sizeof( DrawStatistics ),
									   renderer::Buffer::CpuUsage::READ_WRITE );
	}
}

void SceneRenderer::add_mesh( mesh_handle mesh, glm::vec3 translation, glm::quat rotation, glm::vec3 scale, material_handle material )
{
	_scene.push_back( MeshDraw { .translation = translation, .mesh = mesh, .rotation = rotation, .scale = scale, .material = material } );
	_needs_rebuild = true;
}

void SceneRenderer::clear_meshes()
{
	_scene.clear();
	_needs_rebuild = true;
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
							bool use_pbr,
							bool use_occlusion_culling )
{
	const auto frame_index = _frames % renderer::MAX_FRAMES_IN_FLIGHT;

	if ( _frames >= renderer::MAX_FRAMES_IN_FLIGHT )
	{
		_draw_statistics_buffers[ frame_index ].flush();
		memcpy( &_statistics, _draw_statistics_buffers[ frame_index ].get_mapped_address(), sizeof( DrawStatistics ) );
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

	{
		auto constants = build_scene_constants( camera, _extent, main_light, sky_illumination );
		constants.pbr = use_pbr ? 1 : 0;
		memcpy( static_cast<scene_constants*>( _constant_buffers.get_mapped_address() ) + frame_index, &constants, sizeof( constants ) );
	}
	const auto constants_address = _constant_buffers.get_device_address() + ( frame_index * sizeof( scene_constants ) );

	using Draws = std::array<MeshDraw, MAX_DRAWS>;
	memcpy( static_cast<Draws*>( _draw_buffers.get_mapped_address() ) + frame_index, _scene.data(), sizeof( MeshDraw ) * _scene.size() );
	const auto draw_buffer_address = _draw_buffers.get_device_address() + ( frame_index * sizeof( Draws ) );

	cmd.texture_barrier( _color_texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::COLOR_ATTACHMENT );
	cmd.texture_barrier( _depth_texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::DEPTH_ATTACHMENT );
	cmd.texture_barrier( _resolved_depth_texture.texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::DEPTH_ATTACHMENT );

	cmd.buffer_barrier( _draw_statistics_buffers[ frame_index ],
						renderer::ResourceState::UNDEFINED,
						renderer::ResourceState::TRANSFER_DST );
	cmd.fill_buffer( _draw_statistics_buffers[ frame_index ], 0, sizeof( DrawStatistics ), 0 );
	cmd.buffer_barrier( _draw_statistics_buffers[ frame_index ],
						renderer::ResourceState::TRANSFER_DST,
						renderer::ResourceState::STORAGE_WRITE );

	const auto draw_meshes = [ & ]( bool early )
	{
		//cmd.write_timestamp( timestamp_queries[ image_index ], early ? Timestamps::COMPUTE_DRAWS_EARLY : Timestamps::COMPUTE_DRAWS_LATE );
		{
			PROFILER_GPU_EVENT( "Compute Draws" );
			cmd.buffer_barrier( _draw_commands_buffer, renderer::ResourceState::INDIRECT_ARGUMENT, renderer::ResourceState::TRANSFER_DST );
			cmd.fill_buffer( _draw_commands_buffer, 0, sizeof( uint32_t ), 0 );
			cmd.buffer_barrier( _draw_commands_buffer, renderer::ResourceState::TRANSFER_DST, renderer::ResourceState::STORAGE_READ_WRITE );
			const auto pipeline = early ? pipelines.get( _compute_draws_early_pipeline ) : pipelines.get( _compute_draws_late_pipeline );
			cmd.bind_pipeline( pipeline, bindless_manager );
			const compute_draws_constants constants { .draws_buffer = draw_buffer_address,
													  .commands_buffer = _draw_commands_buffer.get_device_address() + sizeof( uint32_t ),
													  .commmands_count_buffer = _draw_commands_buffer.get_device_address(),
													  .visibility_buffer = _draw_visibility_buffer.get_device_address(),
													  .statistics_buffer = _draw_statistics_buffers[ frame_index ].get_device_address(),
													  .scene_constants = constants_address,
													  .draws_count = static_cast<uint32_t>( _scene.size() ),
													  .lod_target = lod_pixel_tolerance
														  / ( _extent.height / ( 2 * std::tan( 70.f * 0.5f ) ) ),
													  .depth_pyramid = _depth_pyramid.handles.sampled_texture };
			cmd.push_constants( pipeline, constants );
			cmd.dispatch( calc_group_count( static_cast<uint32_t>( _scene.size() ) ), 1, 1 );
			cmd.buffer_barrier( _draw_commands_buffer, renderer::ResourceState::STORAGE_WRITE, renderer::ResourceState::INDIRECT_ARGUMENT );
			cmd.buffer_barrier( _draw_visibility_buffer,
								renderer::ResourceState::STORAGE_READ_WRITE,
								renderer::ResourceState::STORAGE_READ_WRITE );
		}

		renderer::RenderAttachment color_attachment { .target = _color_texture_view };
		if ( early )
		{
			color_attachment.clear_value = { { 0.1f, 0.1f, 0.1f, 1.f } };
		}
		renderer::RenderAttachment depth_attachment { .target = _depth_texture_view };
		if ( early )
		{
			depth_attachment.clear_value = { { 0.f } };
			depth_attachment.resolve_target = _resolved_depth_texture.handles.view;
		}

		cmd.begin_rendering( { .color_target = color_attachment, .depth_target = depth_attachment } );

		cmd.set_viewport( _extent );
		cmd.set_scissor( _extent );

		//cmd.write_timestamp( timestamp_queries[ image_index ], early ? Timestamps::DRAW_MESHES_EARLY : Timestamps::DRAW_MESHES_LATE );
		if ( !index_buffer )
		{
			PROFILER_GPU_EVENT( "Draw Meshlets" );
			cmd.bind_pipeline( pipelines.get( _meshlet_pipeline ), bindless_manager );
			const draw_meshes_constants constants { .scene_constants = constants_address,
													.draws_buffer = draw_buffer_address,
													.commands_buffer = _draw_commands_buffer.get_device_address() + sizeof( uint32_t ) };
			cmd.push_constants( pipelines.get( _meshlet_pipeline ), constants );
			cmd.draw_mesh_tasks_indirect( _draw_commands_buffer,
										  sizeof( uint32_t ) + offsetof( gpu_draw_command, mesh_cmd ),
										  _draw_commands_buffer,
										  0,
										  static_cast<uint32_t>( _scene.size() ),
										  sizeof( gpu_draw_command ) );
		}
		else
		{
			PROFILER_GPU_EVENT( "Draw Meshes" );
			cmd.bind_pipeline( pipelines.get( _mesh_pipeline ), bindless_manager );
			cmd.bind_index_buffer( *index_buffer );
			const draw_meshes_constants constants { .scene_constants = constants_address,
													.draws_buffer = draw_buffer_address,
													.commands_buffer = _draw_commands_buffer.get_device_address() + sizeof( uint32_t ) };

			cmd.push_constants( pipelines.get( _mesh_pipeline ), constants );
			if ( _draw_indirect_count )
			{
				cmd.draw_indexed_indirect( _draw_commands_buffer,
										   sizeof( uint32_t ) + offsetof( gpu_draw_command, draw_cmd ),
										   _draw_commands_buffer,
										   0,
										   static_cast<uint32_t>( _scene.size() ),
										   sizeof( gpu_draw_command ) );
			}
			else
			{
				cmd.draw_indexed_indirect( _draw_commands_buffer,
										   sizeof( uint32_t ) + offsetof( gpu_draw_command, draw_cmd ),
										   static_cast<uint32_t>( _scene.size() ),
										   sizeof( gpu_draw_command ) );
			}
		}
		cmd.end_rendering();
	};

	draw_meshes( true );

	if ( use_occlusion_culling )
	{
		cmd.texture_barrier( _resolved_depth_texture.texture,
							 renderer::ResourceState::DEPTH_ATTACHMENT,
							 renderer::ResourceState::COMPUTE_SHADER_READ );

		{
			//cmd.write_timestamp( timestamp_queries[ image_index ], Timestamps::DEPTH_PYRAMID );
			PROFILER_GPU_EVENT( "Depth Pyramid" );
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

		draw_meshes( false );
	}

	cmd.buffer_barrier( _draw_statistics_buffers[ frame_index ],
						renderer::ResourceState::STORAGE_WRITE,
						renderer::ResourceState::HOST_READ );

	{
		PROFILER_GPU_EVENT( "Draw Grid" );
		cmd.texture_barrier( _resolved_color_texture.texture,
							 renderer::ResourceState::UNDEFINED,
							 renderer::ResourceState::COLOR_ATTACHMENT );
		cmd.begin_rendering( { .color_target = { .target = _color_texture_view, .resolve_target = _resolved_color_texture.handles.view },
							   .depth_target = { .target = _depth_texture_view } } );

		cmd.set_viewport( _extent );
		cmd.set_scissor( _extent );
		//cmd.write_timestamp( timestamp_queries[ image_index ], Timestamps::DRAW_GRID );
		cmd.bind_pipeline( pipelines.get( _grid_pipeline ), bindless_manager );
		cmd.push_constants( pipelines.get( _grid_pipeline ), grid_constants { .scene_constants = constants_address } );
		cmd.draw( 4 );
		cmd.end_rendering();
	}

	{
		PROFILER_GPU_EVENT( "Tonemap" );
		//cmd.write_timestamp( timestamp_queries[ image_index ], Timestamps::BLIT );
		cmd.texture_barrier( _resolved_color_texture.texture,
							 renderer::ResourceState::COLOR_ATTACHMENT,
							 renderer::ResourceState::FRAGMENT_SHADER_READ );
		cmd.texture_barrier( output_texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::COLOR_ATTACHMENT );
		cmd.begin_rendering( { .color_target = { .target = output_texture_view } } );
		cmd.set_viewport( output_texture.get_extent() );
		cmd.set_scissor( output_texture.get_extent() );
		cmd.bind_pipeline( pipelines.get( _tonemap_pipeline ), bindless_manager );
		cmd.push_constants( pipelines.get( _tonemap_pipeline ), _resolved_color_texture.handles.sampled_texture );
		cmd.draw( 3 );
		cmd.end_rendering();
	}

	++_frames;
}
