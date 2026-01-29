#include <SDL3/SDL.h>
#include <array>
#include <chrono>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_aligned.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <memory>
#include <numbers>
#include <numeric>
#include <optick.h>
#include <renderer/third_party/tbb.h>
#include <span>

//
#include <gpu_stable_power/gpu_stable_power.h>
#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/command_buffer.h>
#include <renderer/device.h>
#include <renderer/pipeline_manager.h>
#include <renderer/shader.h>
#include <renderer/swapchain.h>
#include <renderer/texture.h>

//

#include "asset.h"

struct gpu_mesh_draw
{
	glm::aligned_vec3 position;
	mesh_handle mesh;
	material_handle material;
};

// Not used on CPU side, only there for buffer sizing
struct gpu_draw_command
{
	uint32_t draw_id;
	uint32_t lod;
	VkDrawIndexedIndirectCommand draw_cmd;
	VkDrawMeshTasksIndirectCommandEXT mesh_cmd;
};

struct draw_meshes_constants
{
	renderer::BufferHandle scene_constants;
	renderer::BufferHandle draws_buffer;
	renderer::BufferHandle commands_buffer;
};

static constexpr uint32_t MAX_MESHES = 8192;
static constexpr uint32_t MAX_MATERIALS = 8192;

class Camera
{
public:
	glm::vec3 velocity;
	glm::vec3 position;
	float pitch = 0.f;
	float yaw = 0.f;

	glm::mat4 getViewMatrix() const
	{
		glm::mat4 cameraTranslation = glm::translate( glm::mat4( 1.f ), position );
		glm::mat4 cameraRotation = getRotationMatrix();
		return glm::inverse( cameraTranslation * cameraRotation );
	}
	glm::mat4 getRotationMatrix() const
	{
		glm::quat pitchRotation = glm::angleAxis( pitch, glm::vec3 { 1.f, 0.f, 0.f } );
		glm::quat yawRotation = glm::angleAxis( yaw, glm::vec3 { 0.f, -1.f, 0.f } );

		return glm::mat4_cast( yawRotation ) * glm::mat4_cast( pitchRotation );
	}

	void processSDLEvent( SDL_Event& e )
	{
		if ( e.type == SDL_EVENT_KEY_DOWN || SDL_EVENT_KEY_UP )
		{
			const float factor = e.type == SDL_EVENT_KEY_DOWN ? 5.f : 0.f;
			if ( e.key.key == SDLK_W )
			{
				velocity.z = -1 * factor;
			}
			if ( e.key.key == SDLK_S )
			{
				velocity.z = 1 * factor;
			}
			if ( e.key.key == SDLK_A )
			{
				velocity.x = -1 * factor;
			}
			if ( e.key.key == SDLK_D )
			{
				velocity.x = 1 * factor;
			}
		}

		if ( e.type == SDL_EVENT_MOUSE_MOTION )
		{
			yaw += (float)e.motion.xrel / 500.f;
			pitch -= (float)e.motion.yrel / 500.f;
		}
	}

	void update( float delta_time )
	{
		glm::mat4 cameraRotation = getRotationMatrix();
		position += glm::vec3( cameraRotation * glm::vec4( velocity, 0.f ) ) * delta_time;
	}
};

struct tbb_observer final : public tbb::task_scheduler_observer
{
	void on_scheduler_entry( bool is_worker ) override
	{
		if ( is_worker )
		{
			static thread_local OPTICK_THREAD( "TBB Worker" );
		}
	}
};

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
};

glm::vec4 normalize_plane( const glm::vec4& plane )
{
	return plane / glm::length( glm::vec3( plane ) );
}

scene_constants build_scene_constants( const Camera& camera, renderer::Extent2D extent, float light_time )
{
	const glm::mat4 view = camera.getViewMatrix();
	glm::mat4 projection = glm::perspective( glm::radians( 70.f ), static_cast<float>( extent.width ) / extent.height, 10000.f, 0.1f );

	// invert the Y direction on projection matrix so that we are more similar
	// to opengl and gltf axis
	projection[ 1 ][ 1 ] *= -1;

	const glm::vec4 main_light( 20.f * std::cosf( light_time * std::numbers::pi * 2.f ),
								2.f,
								10.f * std::sinf( light_time * std::numbers::pi * 2.f ),
								2.5f );
	scene_constants constants { .view_projection_matrix = projection * view,
								.projection_matrix = projection,
								.view_matrix = view,
								.inverse_view_matrix = glm::inverse( view ),
								.camera_position = camera.position,
								.main_light = main_light };

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

struct ui_data
{
	int x_miku = 1;
	int z_miku = 1;
	float lod_cutoff = 10.f;
	bool paused = false;
	bool vsync;
	bool stable_power;
	bool mesh_shading;
	bool occlusion_culling = true;
	bool pbr = true;
};

namespace Timestamps
{
	enum : uint32_t
	{
		FRAME_START = 0,
		COMPUTE_DRAWS_EARLY,
		DRAW_MESHES_EARLY,
		DEPTH_PYRAMID,
		COMPUTE_DRAWS_LATE,
		DRAW_MESHES_LATE,
		DRAW_GRID,
		DRAW_UI,
		BLIT,
		FRAME_END,
		COUNT
	};
}

struct Timings
{
	explicit Timings( float gpu_timestamp_period )
		: period( gpu_timestamp_period )
	{
	}

	void add( std::span<const uint64_t, Timestamps::COUNT> values, int frame_index )
	{
		using namespace Timestamps;
		const int index = frame_index % last_values[ 0 ].size();
		last_values[ 0 ][ index ] = ( values[ FRAME_END ] - values[ FRAME_START ] ) * period;
		for ( int i = 1; i < last_values.size(); ++i )
		{
			last_values[ i ][ index ] = ( values[ i + 1 ] - values[ i ] ) * period;
		}
		for ( int i = 0; i < last_values.size(); ++i )
		{
			averages[ i ] = std::accumulate( begin( last_values[ i ] ), end( last_values[ i ] ), 0.f ) / last_values[ i ].size();
		}
	}

	std::array<std::array<float, 60>, Timestamps::FRAME_END> last_values;
	std::array<float, Timestamps::FRAME_END> averages;
	float period;
};

void draw_ui( ui_data& data,
			  const renderer::Device::Properties& props,
			  const BindlessManager& bindless_manager,
			  const index_buffer& index_buffer,
			  std::span<const asset> assets,
			  std::span<const float> gpu_timings,
			  const renderer::Statistics& stats )
{
	OPTICK_EVENT();
	ImGui::Begin( "Vulkan renderer" );
	{
		ImGui::SeparatorText( "Hardware" );
		ImGui::Text( "GPU: %s", props.name.c_str() );
		ImGui::Text( "Memory (MB): RAM: %zu, VRAM: %zu, BAR: %zu",
					 props.host_memory_size >> 20,
					 props.device_memory_size >> 20,
					 props.transfer_memory_size >> 20 );
		ImGui::SeparatorText( "VRAM usage" );
		ImGui::Text( "Textures (MB): %zu", bindless_manager.get_texture_memory_usage() >> 20 );
		size_t meshes_bytes = 0;
		for ( const auto& asset : assets )
		{
			for ( const auto& mesh : asset.meshes )
			{
				meshes_bytes += mesh.data_buffer.get_size();
			}
		}
		ImGui::Text( "Meshes (MB): %zu", meshes_bytes >> 20 );
		ImGui::Text( "Indices (MB): %zu", ( index_buffer.size * sizeof( uint32_t ) ) >> 20 );

		std::array<float, 4> per_lod_gpu_time {};
		ImGui::SeparatorText( "Stats" );
		ImGui::Text( "GPU times (ms): %.2f", gpu_timings[ 0 ] * 1e-6f );
		ImGui::BulletText( "Early meshes draw: %.2f / %.2f",
						   gpu_timings[ Timestamps::COMPUTE_DRAWS_EARLY ] * 1e-6f,
						   gpu_timings[ Timestamps::DRAW_MESHES_EARLY ] * 1e-6f );
		ImGui::BulletText( "Late meshes draw: %.2f / %.2f",
						   gpu_timings[ Timestamps::COMPUTE_DRAWS_LATE ] * 1e-6f,
						   gpu_timings[ Timestamps::DRAW_MESHES_LATE ] * 1e-6f );
		ImGui::BulletText( "Depth pyramid: %.2f", gpu_timings[ Timestamps::DEPTH_PYRAMID ] * 1e-6f );
		ImGui::BulletText( "UI: %.2f", gpu_timings[ Timestamps::DRAW_UI ] * 1e-6f );
		ImGui::BulletText( "Triangles (millions): %.2f", static_cast<float>( stats.clipping_invocations / 1000 ) / 1000 );
		ImGui::SeparatorText( "Assets" );
		for ( int i = 0; i < assets.size(); ++i )
		{
			for ( int j = 0; j < assets[ i ].meshes.size(); ++j )
			{
				ImGui::Text( "Asset #%d Mesh #%d: %d / %d / %d / %d triangles",
							 i,
							 j,
							 assets[ i ].meshes[ j ].gpu_data.lods[ 0 ].indices_count,
							 assets[ i ].meshes[ j ].gpu_data.lods[ 1 ].indices_count,
							 assets[ i ].meshes[ j ].gpu_data.lods[ 2 ].indices_count,
							 assets[ i ].meshes[ j ].gpu_data.lods[ 3 ].indices_count );
			}
		}

		ImGui::SeparatorText( "Settings" );
		if ( ImGui::Button( data.paused ? "Unpause" : "Pause" ) )
		{
			data.paused = !data.paused;
		}
		ImGui::Checkbox( "V-Sync", &data.vsync );
		ImGui::Checkbox( "Stable power", &data.stable_power );
		ImGui::Checkbox( "Mesh shading", &data.mesh_shading );
		ImGui::Checkbox( "Occlusion culling", &data.occlusion_culling );
		ImGui::Checkbox( "PBR", &data.pbr );
		ImGui::Text( "Instance count" );
		ImGui::SliderInt( "x", &data.x_miku, 1, 100 );
		ImGui::SliderInt( "z", &data.z_miku, 1, 100 );
		ImGui::SliderFloat( "LOD cutoff", &data.lod_cutoff, 0.f, 200.f );
	}
	ImGui::End();
}

struct grid_constants
{
	glm::aligned_vec3 grid_color = { 0.1f, 0.1f, 0.1f };
	renderer::BufferHandle scene_constants;
	float near_distance = 50.f;
	float far_distance = 60.f;
};

struct depth_pyramid_constants
{
	uint32_t depth_in;
	uint32_t depth_out;
	glm::aligned_vec2 size;
};

struct compute_draws_constants
{
	renderer::BufferHandle draws_buffer;
	renderer::BufferHandle commands_buffer;
	renderer::BufferHandle commmands_count_buffer;
	renderer::BufferHandle visibility_buffer;
	renderer::BufferHandle scene_constants;
	uint32_t draws_count;
	float lod_cutoff;
	uint32_t depth_pyramid;
};

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

int calc_mip_levels( renderer::Extent2D extent )
{
	int mips = 1;

	while ( extent.width > 1 || extent.height > 1 )
	{
		++mips;
		extent.width /= 2;
		extent.height /= 2;
	}

	return mips;
}

int main()
{
	constexpr bool profile_startup = true;

	if ( profile_startup )
	{
		OPTICK_START_CAPTURE();
	}

	OPTICK_PUSH( "Startup" );

	tbb_observer observer {};
	observer.observe();

	gpu_stable_power::Context stable_power;

	OPTICK_PUSH( "create_device" );
	renderer::Device device( "Brutus-NG [Vulkan]" );
	const auto props = device.get_properties();
	OPTICK_POP();

	bool vsync = true;

	OPTICK_PUSH( "create_swapchain" );
	constexpr auto swapchain_image_format = renderer::Texture::Format::R8G8B8A8_SRGB;
	renderer::Swapchain swapchain( device, swapchain_image_format, vsync );
	OPTICK_POP();

	constexpr auto draw_image_format = renderer::Texture::Format::R16G16B16A16_SFLOAT;

	{
		OPTICK_EVENT( "init_imgui" );
		ImGui::CreateContext();
		const auto internals = device.get_internals();
		ImGui_ImplSDL3_InitForVulkan( internals.window );
		const VkFormat color_format = static_cast<VkFormat>( draw_image_format );
		ImGui_ImplVulkan_InitInfo info { .ApiVersion = internals.api_version,
										 .Instance = internals.instance,
										 .PhysicalDevice = internals.physical_device,
										 .Device = internals.device,
										 .QueueFamily = internals.queue_family,
										 .Queue = internals.queue,
										 .MinImageCount = swapchain.get_image_count(),
										 .ImageCount = swapchain.get_image_count(),
										 .DescriptorPoolSize = renderer::MAX_FRAMES_IN_FLIGHT,
										 .UseDynamicRendering = true,
										 .PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
																		  .colorAttachmentCount = 1,
																		  .pColorAttachmentFormats = &color_format } };
		ImGui_ImplVulkan_Init( &info );
	}

	const auto draw_image_extent = device.get_extent();
	using usage = renderer::Texture::Usage;
	auto draw_image = device.create_texture( renderer::Texture::Desc { .format = draw_image_format,
																	   .usage = usage::COLOR_ATTACHMENT,
																	   .extent = draw_image_extent,
																	   .samples = 4 } );
	auto draw_image_view = device.create_texture_view( draw_image, renderer::TextureView::Aspect::COLOR );

	auto depth_image = device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::D32_SFLOAT,
																		.usage = usage::DEPTH_STENCIL_ATTACHMENT,
																		.extent = draw_image_extent,
																		.samples = 4 } );

	auto depth_image_view = device.create_texture_view( depth_image, renderer::TextureView::Aspect::DEPTH );

	auto draw_resolved_image = device.create_texture( renderer::Texture::Desc { .format = draw_image_format,
																				.usage = usage::COLOR_ATTACHMENT | usage::TRANSFER_SRC,
																				.extent = draw_image_extent } );
	auto draw_resolved_image_view = device.create_texture_view( draw_resolved_image, renderer::TextureView::Aspect::COLOR );

	BindlessManager bindless_manager( device, std::array { MAX_MESHES, MAX_MATERIALS } );

	const auto depth_resolved_image = bindless_manager.add_texture(
		device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::D32_SFLOAT,
														 .usage = usage::DEPTH_STENCIL_ATTACHMENT | usage::SAMPLED,
														 .extent = draw_image_extent } ) );

	const renderer::Extent2D depth_pyramid_extent { round_pow2( draw_image_extent.width ), round_pow2( draw_image_extent.height ) };
	const auto depth_pyramid = bindless_manager.add_texture(
		device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::R32_SFLOAT,
														 .usage = usage::SAMPLED | usage::STORAGE,
														 .extent = depth_pyramid_extent,
														 .mips = calc_mip_levels( depth_pyramid_extent ) } ),
		true );

	index_buffer indices( device, 1024 * 1024 );

	renderer::PipelineManager pipelines( device, "../shaders/vulkan", bindless_manager );

	const auto mesh_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = draw_image_format,
								   .depth_format = depth_image.get_format(),
								   .push_constants_size = sizeof( draw_meshes_constants ) },
		{ { "mesh.vert", renderer::ShaderStage::VERTEX }, { "mesh.frag", renderer::ShaderStage::FRAGMENT } } );

	const auto mesh_pipeline_meshlet = pipelines.add( renderer::Pipeline::Desc { .color_format = draw_image_format,
																				 .depth_format = depth_image.get_format(),
																				 .push_constants_size = sizeof( draw_meshes_constants ) },
													  { { "mesh.task", renderer::ShaderStage::TASK },
														{ "mesh.mesh", renderer::ShaderStage::MESH },
														{ "mesh.frag", renderer::ShaderStage::FRAGMENT } } );

	const auto compute_draws_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .push_constants_size = sizeof( compute_draws_constants ) },
		{ { "mesh_cull.comp", renderer::ShaderStage::COMPUTE, { { "LATE", "false" } } } } );

	const auto compute_draws_late_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .push_constants_size = sizeof( compute_draws_constants ) },
		{ { "mesh_cull.comp", renderer::ShaderStage::COMPUTE, { { "LATE", "true" } } } } );

	const auto depth_pyramid_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .push_constants_size = sizeof( depth_pyramid_constants ) },
		{ { "depth_pyramid.comp", renderer::ShaderStage::COMPUTE } } );

	const auto grid_pipeline = pipelines.add(
		renderer::Pipeline::Desc { .color_format = draw_image_format,
								   .depth_format = depth_image.get_format(),
								   .topology = renderer::Pipeline::PrimitiveTopology::TRIANGLE_STRIP,
								   .push_constants_size = sizeof( grid_constants ) },
		{ { "grid.vert", renderer::ShaderStage::VERTEX }, { "grid.frag", renderer::ShaderStage::FRAGMENT } } );

	std::vector<asset> assets;
	assets.push_back( load_asset( device, bindless_manager, indices, "../assets/player.glb" ) );
	assets.push_back( load_asset( device, bindless_manager, indices, "../assets/sabrina.glb" ) );

	auto constant_buffers = device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
												  sizeof( std::array<scene_constants, renderer::MAX_FRAMES_IN_FLIGHT> ),
												  true );
	const auto constant_buffers_ptr = static_cast<scene_constants*>( constant_buffers.get_mapped_address() );

	static const uint32_t max_draws = 1024 * 1024;
	using Draws = std::array<gpu_mesh_draw, max_draws>;
	auto draw_buffers = device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
											  sizeof( Draws ) * renderer::MAX_FRAMES_IN_FLIGHT,
											  true );
	const auto draw_buffers_ptr = static_cast<Draws*>( draw_buffers.get_mapped_address() );

	using DrawVisibility = std::array<uint8_t, max_draws>;
	auto draw_visibility_buffer = device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
															| renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
														sizeof( DrawVisibility ) );
	struct DrawCommands
	{
		uint32_t count;
		std::array<gpu_draw_command, max_draws> commands;
	};
	auto draw_commands_buffer = device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::INDIRECT_BUFFER
														  | renderer::Buffer::Usage::TRANSFER_DST
														  | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
													  sizeof( DrawCommands ) );

	std::array<renderer::raii::TimestampQuery, renderer::MAX_FRAMES_IN_FLIGHT> timestamp_queries {
		device.create_timestamp_query( Timestamps::COUNT ),
		device.create_timestamp_query( Timestamps::COUNT )
	};
	std::array<renderer::raii::StatisticsQuery, renderer::MAX_FRAMES_IN_FLIGHT> statistics_queries { device.create_statistics_query(),
																									 device.create_statistics_query() };

	pipelines.wait_ready();

	OPTICK_POP();

	Camera camera { .position = glm::vec3( 0, 1.f, 2.f ) };
	Timings gpu_timings( device.get_timestamp_period() );

	if ( profile_startup )
	{
		OPTICK_STOP_CAPTURE();
#ifdef _DEBUG
		OPTICK_SAVE_CAPTURE( "startup_debug.opt" );
#else
		OPTICK_SAVE_CAPTURE( "startup_release.opt" );
#endif
	}

	bool quit = false;
	int frame_count = 0;

	auto time = std::chrono::steady_clock::now();
	constexpr float day_period = 10.f;
	float time_of_day = 0.f;

	bool relative_mouse = false;
	ui_data ui_state { .vsync = vsync, .stable_power = stable_power.is_enabled(), .mesh_shading = props.mesh_shader_support };
	uint64_t last_scene_hash = 0;

	while ( !quit )
	{
		OPTICK_FRAME( "MainThread" );
		SDL_Event event;
		while ( SDL_PollEvent( &event ) )
		{
			if ( event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED
				 || ( event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE ) )
			{
				quit = true;
			}
			else if ( event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_LALT )
			{
				relative_mouse = !relative_mouse;
				device.set_relative_mouse_mode( relative_mouse );
			}
			if ( relative_mouse )
			{
				camera.processSDLEvent( event );
			}
			else
			{
				ImGui_ImplSDL3_ProcessEvent( &event );
			}
		}
		const auto old_time = time;
		time = std::chrono::steady_clock::now();
		using namespace std::chrono_literals;
		const auto delta_time = duration_cast<std::chrono::microseconds>( time - old_time ).count()
			/ static_cast<float>( duration_cast<std::chrono::microseconds>( 1s ).count() );
		if ( !ui_state.paused )
		{
			time_of_day += delta_time;
			if ( time_of_day > day_period )
			{
				time_of_day -= day_period;
			}
		}
		camera.update( delta_time );

		const bool mesh_shading = ui_state.mesh_shading;
		const bool occlusion_culling = ui_state.occlusion_culling;

		auto [ image_index, swapchain_image, swapchain_image_view ] = swapchain.acquire();

		const uint32_t query_pool_offset = image_index * Timestamps::COUNT;
		renderer::Statistics stats;
		if ( frame_count > renderer::MAX_FRAMES_IN_FLIGHT )
		{
			std::array<uint64_t, Timestamps::COUNT> timestamps;
			device.get_query_results( timestamp_queries[ image_index ], 0, timestamps );
			gpu_timings.add( timestamps, frame_count );
			stats = device.get_query_results( statistics_queries[ image_index ] );
		}

		{
			auto constants = build_scene_constants( camera, draw_image_extent, time_of_day / day_period );
			constants.pbr = ui_state.pbr ? 1 : 0;
			memcpy( constant_buffers_ptr + image_index, &constants, sizeof( constants ) );
		}
		const auto constants_address = constant_buffers.get_device_address() + ( image_index * sizeof( scene_constants ) );

		auto draw_buffer = ( draw_buffers_ptr + image_index )->data();
		uint32_t draw_count = 0;
		std::minstd_rand rand;
		std::uniform_int_distribution gen( 0zu, assets.size() - 1 );

		for ( int z = 0; z < ui_state.z_miku; ++z )
		{
			for ( int x = 0; x < ui_state.x_miku; ++x )
			{
				const auto pos = glm::vec3( x - ( ui_state.x_miku / 2 ), 0, z - ( ui_state.z_miku / 2 ) );
				const auto asset = gen( rand );
				for ( int i = 0; i < assets[ asset ].meshes.size(); ++i )
				{
					draw_buffer[ draw_count ].position = pos;
					draw_buffer[ draw_count ].mesh = assets[ asset ].meshes[ i ].handle;
					draw_buffer[ draw_count ].material = assets[ asset ].materials[ i ].handle;
					++draw_count;
				}
			}
		}
		const uint64_t scene_hash = static_cast<uint64_t>( ui_state.z_miku ) << 32 | ui_state.x_miku;
		const auto draw_buffer_address = draw_buffers.get_device_address() + ( image_index * sizeof( Draws ) );

		pipelines.update();

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();

		draw_ui( ui_state, props, bindless_manager, indices, assets, gpu_timings.averages, stats );

		auto command_buffer = device.grab_command_buffer();
		command_buffer->reset();
		command_buffer->begin();

		command_buffer->reset_query( timestamp_queries[ image_index ], 0, Timestamps::COUNT );
		command_buffer->write_timestamp( timestamp_queries[ image_index ], Timestamps::FRAME_START );

		command_buffer->reset_query( statistics_queries[ image_index ] );
		command_buffer->begin_query( statistics_queries[ image_index ] );

		if ( last_scene_hash != scene_hash || !occlusion_culling )
		{
			// Can't reuse visibility buffer if the scene has changed
			command_buffer->fill_buffer( draw_visibility_buffer, 0, draw_visibility_buffer.get_size(), 0 );
			command_buffer->buffer_barrier( draw_visibility_buffer );
			last_scene_hash = scene_hash;
		}

		command_buffer->transition_texture( draw_image,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( draw_resolved_image,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( depth_image,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::DEPTH_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( depth_resolved_image.texture,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::DEPTH_ATTACHMENT_OPTIMAL );

		const auto draw_meshes = [ & ]( bool early )
		{
			command_buffer->write_timestamp( timestamp_queries[ image_index ],
											 early ? Timestamps::COMPUTE_DRAWS_EARLY : Timestamps::COMPUTE_DRAWS_LATE );
			{
				OPTICK_GPU_EVENT( "Compute Draws" );
				command_buffer->buffer_barrier( draw_commands_buffer );
				command_buffer->fill_buffer( draw_commands_buffer, 0, sizeof( uint32_t ), 0 );
				command_buffer->buffer_barrier( draw_commands_buffer );
				const auto pipeline = early ? pipelines.get( compute_draws_pipeline ) : pipelines.get( compute_draws_late_pipeline );
				command_buffer->bind_pipeline( pipeline, bindless_manager );
				const compute_draws_constants constants { .draws_buffer = draw_buffer_address,
														  .commands_buffer = draw_commands_buffer.get_device_address() + sizeof( uint32_t ),
														  .commmands_count_buffer = draw_commands_buffer.get_device_address(),
														  .visibility_buffer = draw_visibility_buffer.get_device_address(),
														  .scene_constants = constants_address,
														  .draws_count = draw_count,
														  .lod_cutoff = ui_state.lod_cutoff,
														  .depth_pyramid = depth_pyramid.handles.texture_index };
				command_buffer->push_constants( pipeline, constants );
				command_buffer->dispatch( calc_group_count( draw_count ), 1, 1 );
				command_buffer->buffer_barrier( draw_commands_buffer );
				command_buffer->buffer_barrier( draw_visibility_buffer );
			}

			renderer::RenderAttachment color_attachment { .target = draw_image_view };
			if ( early )
			{
				color_attachment.clear_value = { { 0.1f, 0.1f, 0.1f, 1.f } };
			}
			renderer::RenderAttachment depth_attachment { .target = depth_image_view };
			if ( early )
			{
				depth_attachment.clear_value = { { 0.f } };
				depth_attachment.resolve_target = depth_resolved_image.handles.view;
			}

			command_buffer->begin_rendering( draw_image_extent, color_attachment, depth_attachment );

			command_buffer->set_viewport( draw_image_extent );
			command_buffer->set_scissor( draw_image_extent );

			command_buffer->write_timestamp( timestamp_queries[ image_index ],
											 early ? Timestamps::DRAW_MESHES_EARLY : Timestamps::DRAW_MESHES_LATE );
			if ( mesh_shading )
			{
				OPTICK_GPU_EVENT( "Draw Meshlets" );
				command_buffer->bind_pipeline( pipelines.get( mesh_pipeline_meshlet ), bindless_manager );
				const draw_meshes_constants constants { .scene_constants = constants_address,
														.draws_buffer = draw_buffer_address,
														.commands_buffer = draw_commands_buffer.get_device_address() + sizeof( uint32_t ) };
				command_buffer->push_constants( pipelines.get( mesh_pipeline_meshlet ), constants );
				command_buffer->draw_mesh_tasks_indirect( draw_commands_buffer,
														  sizeof( uint32_t ) + offsetof( gpu_draw_command, mesh_cmd ),
														  draw_commands_buffer,
														  0,
														  max_draws,
														  sizeof( gpu_draw_command ) );
			}
			else
			{
				OPTICK_GPU_EVENT( "Draw Meshes" );
				command_buffer->bind_pipeline( pipelines.get( mesh_pipeline ), bindless_manager );
				command_buffer->bind_index_buffer( indices.buffer );
				const draw_meshes_constants constants { .scene_constants = constants_address,
														.draws_buffer = draw_buffer_address,
														.commands_buffer = draw_commands_buffer.get_device_address() + sizeof( uint32_t ) };

				command_buffer->push_constants( pipelines.get( mesh_pipeline ), constants );
				command_buffer->draw_indexed_indirect( draw_commands_buffer,
													   sizeof( uint32_t ) + offsetof( gpu_draw_command, draw_cmd ),
													   draw_commands_buffer,
													   0,
													   max_draws,
													   sizeof( gpu_draw_command ) );
			}
			command_buffer->end_rendering();
		};

		draw_meshes( true );

		command_buffer->transition_texture( depth_resolved_image.texture,
											renderer::Texture::Layout::DEPTH_ATTACHMENT_OPTIMAL,
											renderer::Texture::Layout::SHADER_READ_ONLY_OPTIMAL );

		{
			command_buffer->write_timestamp( timestamp_queries[ image_index ], Timestamps::DEPTH_PYRAMID );
			OPTICK_GPU_EVENT( "Depth Pyramid" );
			command_buffer->bind_pipeline( pipelines.get( depth_pyramid_pipeline ), bindless_manager );
			auto extent = depth_pyramid_extent;
			for ( int mip = 0; mip < depth_pyramid.mips.size(); ++mip )
			{
				command_buffer->transition_texture( depth_pyramid.texture,
													renderer::Texture::Layout::UNDEFINED,
													renderer::Texture::Layout::GENERAL,
													mip );

				const depth_pyramid_constants constants { .depth_in = mip == 0 ? depth_resolved_image.handles.texture_index
																			   : depth_pyramid.mips[ mip - 1 ].texture_index,
														  .depth_out = depth_pyramid.mips[ mip ].storage_index,
														  .size = glm::aligned_vec2( extent.width, extent.height ) };
				command_buffer->push_constants( pipelines.get( depth_pyramid_pipeline ), constants );
				command_buffer->dispatch( calc_group_count( extent.width ), calc_group_count( extent.height ), 1 );

				command_buffer->transition_texture( depth_pyramid.texture,
													renderer::Texture::Layout::GENERAL,
													renderer::Texture::Layout::SHADER_READ_ONLY_OPTIMAL,
													mip );

				extent.width = std::max( extent.width / 2, 1u );
				extent.height = std::max( extent.height / 2, 1u );
			}
		}

		draw_meshes( false );

		{
			OPTICK_GPU_EVENT( "Draw Grid" );
			command_buffer->begin_rendering(
				draw_image_extent,
				renderer::RenderAttachment { .target = draw_image_view, .resolve_target = draw_resolved_image_view },
				renderer::RenderAttachment { .target = depth_image_view } );

			command_buffer->set_viewport( draw_image_extent );
			command_buffer->set_scissor( draw_image_extent );
			command_buffer->write_timestamp( timestamp_queries[ image_index ], Timestamps::DRAW_GRID );
			command_buffer->bind_pipeline( pipelines.get( grid_pipeline ), bindless_manager );
			command_buffer->push_constants( pipelines.get( grid_pipeline ), grid_constants { .scene_constants = constants_address } );
			command_buffer->draw( 4 );
			command_buffer->end_rendering();
		}

		command_buffer->write_timestamp( timestamp_queries[ image_index ], Timestamps::DRAW_UI );
		command_buffer->begin_rendering( draw_image_extent, renderer::RenderAttachment { .target = draw_resolved_image_view } );

		{
			OPTICK_GPU_EVENT( "Draw UI" );
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData( ImGui::GetDrawData(), command_buffer->get_impl() );
		}
		command_buffer->end_rendering();

		command_buffer->write_timestamp( timestamp_queries[ image_index ], Timestamps::BLIT );

		command_buffer->transition_texture( draw_resolved_image,
											renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL,
											renderer::Texture::Layout::TRANSFER_SRC_OPTIMAL );
		command_buffer->transition_texture( swapchain_image,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::TRANSFER_DST_OPTIMAL );

		command_buffer->blit_texture( draw_resolved_image, swapchain_image );

		command_buffer->transition_texture( swapchain_image,
											renderer::Texture::Layout::TRANSFER_DST_OPTIMAL,
											renderer::Texture::Layout::PRESENT_SRC );

		command_buffer->write_timestamp( timestamp_queries[ image_index ], Timestamps::FRAME_END );
		command_buffer->end_query( statistics_queries[ image_index ] );

		command_buffer->end();

		swapchain.submit( *command_buffer );

		swapchain.present();
		++frame_count;

		if ( ui_state.vsync != vsync )
		{
			vsync = ui_state.vsync;
			swapchain.recreate( swapchain_image_format, vsync );
			ImGui_ImplVulkan_SetMinImageCount( swapchain.get_image_count() );
		}
		ui_state.stable_power = stable_power.set_enabled( ui_state.stable_power );
	}

	// Wait for the GPU to finish before we roll all the destructors
	device.wait_idle();

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();

	OPTICK_SHUTDOWN();

	return 0;
}
