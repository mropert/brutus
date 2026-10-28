#define GLM_ENABLE_EXPERIMENTAL // for matrix decompose

#include <SDL3/SDL.h>
#include <argparse/argparse.hpp>
#include <array>
#include <chrono>
#include <format>
#include <fstream>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_aligned.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <memory>
#include <numbers>
#include <numeric>
#include <print>
#include <random>
#include <renderer/details/profiler.h>
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
#include "camera.h"
#include "gpu_timing.h"
#include "scene_renderer.h"

static constexpr uint32_t MAX_MESHES = 4096;
static constexpr uint32_t MAX_MATERIALS = 8192;

void processSDLEvent( const SDL_Event& e, Camera& camera, glm::vec3& velocity )
{
	if ( e.type == SDL_EVENT_KEY_DOWN || e.type == SDL_EVENT_KEY_UP )
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
		camera.yaw += (float)e.motion.xrel / 500.f;
		camera.pitch -= (float)e.motion.yrel / 500.f;
	}
}

struct tbb_observer final : public tbb::task_scheduler_observer
{
	void on_scheduler_entry( bool is_worker ) override
	{
		if ( is_worker )
		{
			PROFILER_THREAD_STATIC( "TBB Worker" );
		}
	}
};

struct ui_data
{
	int x_miku = 1;
	int z_miku = 1;
	int lod_pixel_tolerance = 1;
	float sun_intensity = 4.5f;
	float sky_to_sun = 0.06f;
	float shadow_distance = 25.f;
	int shadow_cascades = SceneRenderer::SHADOW_CASCADE_MAX;
	bool show_shadow_cascades = false;
	bool paused = false;
	bool vsync;
	bool stable_power;
	bool mesh_shading;
	bool occlusion_culling;
	bool pbr = true;
};

void draw_ui( ui_data& data,
			  const renderer::Device::Properties& props,
			  const renderer::MemoryUsage& memory_usage,
			  const BindlessManager& bindless_manager,
			  std::span<const float> gpu_timings,
			  const renderer::Statistics& stats,
			  const SceneRenderer::DrawStatistics& scene_stats,
			  const Camera& camera )
{
	PROFILER_SCOPE();
	ImGui::Begin( "Vulkan renderer" );
	{
		ImGui::SeparatorText( "Hardware" );
		ImGui::Text( "GPU: %s", props.name.c_str() );
		ImGui::Text( "Memory (MB): RAM: %zu, VRAM: %zu, BAR: %zu",
					 props.host_memory_size >> 20,
					 props.device_memory_size >> 20,
					 props.transfer_memory_size >> 20 );
		ImGui::SeparatorText( "Memory usage" );
		ImGui::Text( "Memory (MB): RAM: %zu, VRAM: %zu, BAR: %zu",
					 memory_usage.ram >> 20,
					 memory_usage.vram >> 20,
					 memory_usage.bar >> 20 );
		const auto resource_mb = [ & ]( renderer::MemoryResource resource )
		{ return memory_usage.resources[ std::to_underlying( resource ) ] >> 20; };
		ImGui::Text( "Textures (MB): %zu (bindless: %zu)",
					 resource_mb( renderer::MemoryResource::TEXTURE ),
					 bindless_manager.get_texture_memory_usage() >> 20 );
		ImGui::Text( "Buffers (MB): storage: %zu, index: %zu, other: %zu",
					 resource_mb( renderer::MemoryResource::STORAGE_BUFFER ),
					 resource_mb( renderer::MemoryResource::INDEX_BUFFER ),
					 resource_mb( renderer::MemoryResource::OTHER_BUFFER ) );

		if ( props.statistics_support )
		{
			ImGui::SeparatorText( "Stats" );
			ImGui::Text( "GPU times (ms): %.2f", gpu_timings[ std::to_underlying( GpuTiming::FRAME ) ] * 1e-6f );
			ImGui::BulletText( "Shadows culling: %.2f", gpu_timings[ std::to_underlying( GpuTiming::CULLING_SHADOWS ) ] * 1e-6f );
			ImGui::BulletText( "Shadows draw: %.2f", gpu_timings[ std::to_underlying( GpuTiming::DRAW_SHADOWS ) ] * 1e-6f );
			ImGui::BulletText( "Early culling: %.2f", gpu_timings[ std::to_underlying( GpuTiming::CULLING_EARLY ) ] * 1e-6f );
			ImGui::BulletText( "Early opaques draw: %.2f", gpu_timings[ std::to_underlying( GpuTiming::DRAW_OPAQUES_EARLY ) ] * 1e-6f );
			ImGui::BulletText( "Late culling: %.2f", gpu_timings[ std::to_underlying( GpuTiming::CULLING_LATE ) ] * 1e-6f );
			ImGui::BulletText( "Late opaques draw: %.2f", gpu_timings[ std::to_underlying( GpuTiming::DRAW_OPAQUES_LATE ) ] * 1e-6f );
			ImGui::BulletText( "Transparents draw: %.2f", gpu_timings[ std::to_underlying( GpuTiming::DRAW_TRANSPARENTS ) ] * 1e-6f );
			ImGui::BulletText( "Depth pyramid: %.2f", gpu_timings[ std::to_underlying( GpuTiming::DEPTH_PYRAMID ) ] * 1e-6f );
			ImGui::BulletText( "Grid draw: %.2f", gpu_timings[ std::to_underlying( GpuTiming::DRAW_GRID ) ] * 1e-6f );
			ImGui::BulletText( "UI: %.2f", gpu_timings[ std::to_underlying( GpuTiming::DRAW_UI ) ] * 1e-6f );
			ImGui::BulletText( "Tonemap: %.2f", gpu_timings[ std::to_underlying( GpuTiming::TONEMAP ) ] * 1e-6f );
			ImGui::BulletText( "Triangles (millions): %.2f", static_cast<float>( stats.clipping_invocations / 1000 ) / 1000 );
		}
		if ( ImGui::BeginTable( "scene_stats", MESH_LOD_COUNT + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg ) )
		{
			ImGui::TableSetupColumn( "LOD" );
			for ( int i = 0; i < MESH_LOD_COUNT; ++i )
			{
				ImGui::TableSetupColumn( std::to_string( i ).c_str() );
			}
			ImGui::TableHeadersRow();
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex( 0 );
			ImGui::Text( "Meshes" );
			for ( int i = 0; i < MESH_LOD_COUNT; ++i )
			{
				ImGui::TableSetColumnIndex( i + 1 );
				ImGui::Text( "%d", scene_stats.meshes[ i ] );
			}
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex( 0 );
			ImGui::Text( "Tris" );
			for ( int i = 0; i < MESH_LOD_COUNT; ++i )
			{
				ImGui::TableSetColumnIndex( i + 1 );
				if ( scene_stats.indices[ i ] / 3 > 100'000 )
				{
					ImGui::Text( "%.2fm", scene_stats.indices[ i ] / 3000000.f );
				}
				else
				{
					ImGui::Text( "%.2fk", scene_stats.indices[ i ] / 3000.f );
				}
			}
			ImGui::EndTable();
		}

		ImGui::SeparatorText( "Camera" );
		ImGui::Text( "Position: %.4f, %.4f, %.4f", camera.position.x, camera.position.y, camera.position.z );
		ImGui::Text( "Pitch: %.4f, yaw: %.4f", camera.pitch, camera.yaw );
		// Same form as the benchmark camera setup, to reproduce a view
		if ( ImGui::Button( "Copy camera" ) )
		{
			const auto setup = std::format(
				"camera.position = {{ {:.4f}f, {:.4f}f, {:.4f}f }};\ncamera.pitch = {:.4f}f;\ncamera.yaw = {:.4f}f;",
				camera.position.x,
				camera.position.y,
				camera.position.z,
				camera.pitch,
				camera.yaw );
			ImGui::SetClipboardText( setup.c_str() );
		}

		ImGui::SeparatorText( "Settings" );
		ImGui::SliderFloat( "Sun intensity", &data.sun_intensity, 1.f, 10.f );
		ImGui::SliderFloat( "Sky intensity", &data.sky_to_sun, 0.01f, 0.3f );
		ImGui::SliderFloat( "Shadow distance", &data.shadow_distance, 5.f, 100.f, "%.3f", ImGuiSliderFlags_AlwaysClamp );
		ImGui::SliderInt( "Shadow cascades", &data.shadow_cascades, 0, SceneRenderer::SHADOW_CASCADE_MAX );
		ImGui::Checkbox( "Show shadow cascades", &data.show_shadow_cascades );
		if ( ImGui::Button( data.paused ? "Unpause" : "Pause" ) )
		{
			data.paused = !data.paused;
		}
		ImGui::Checkbox( "V-Sync", &data.vsync );
		ImGui::Checkbox( "Stable power", &data.stable_power );
		if ( !props.mesh_shader_support )
		{
			ImGui::BeginDisabled();
		}
		ImGui::Checkbox( "Mesh shading", &data.mesh_shading );
		if ( !props.mesh_shader_support )
		{
			ImGui::EndDisabled();
		}
		if ( !props.minmax_filter_support )
		{
			ImGui::BeginDisabled();
		}
		ImGui::Checkbox( "Occlusion culling", &data.occlusion_culling );
		if ( !props.minmax_filter_support )
		{
			ImGui::EndDisabled();
		}
		ImGui::Checkbox( "PBR", &data.pbr );
		ImGui::Text( "Instance count" );
		ImGui::SliderInt( "x", &data.x_miku, 1, 100 );
		ImGui::SliderInt( "z", &data.z_miku, 1, 100 );
		ImGui::SliderInt( "LOD pixel tolerance", &data.lod_pixel_tolerance, 1, 8 );
	}
	ImGui::End();
}

void draw_node( const asset& scene, uint32_t node_index, SceneRenderer& scene_renderer, const glm::mat4& transform )
{
	const auto local_transform = transform * scene.nodes[ node_index ].transform;
	if ( scene.nodes[ node_index ].mesh != size_t( -1 ) )
	{
		const auto& mesh = scene.meshes[ scene.nodes[ node_index ].mesh ];
		glm::vec3 scale, translation, skew;
		glm::vec4 perspective;
		glm::quat rotation;
		glm::decompose( local_transform, scale, rotation, translation, skew, perspective );
		scene_renderer.add_mesh( mesh, translation, rotation, scale, scene.materials[ mesh.material_index ] );
	}
	for ( const auto child : scene.nodes[ node_index ].children )
	{
		draw_node( scene, child, scene_renderer, local_transform );
	}
}

void draw_scene( const asset& scene, SceneRenderer& scene_renderer )
{
	scene_renderer.clear_meshes();
	for ( const auto root : scene.scene_roots )
	{
		draw_node( scene, root, scene_renderer, glm::mat4( 1.f ) );
	}
}

struct crowd_member
{
	size_t asset_index;
	glm::vec3 position;
	glm::quat rotation;
	float scale;
};

// Same crowd on every call: the generator uses its default seed
std::vector<crowd_member> generate_crowd( size_t asset_count, int x_count, int z_count )
{
	std::minstd_rand rand;
	std::uniform_int_distribution gen( 0zu, asset_count - 1 );
	std::uniform_int_distribution gen_angle( 0, 360 );
	std::uniform_real_distribution<float> gen_scale( 0.5f, 1.5f );

	std::vector<crowd_member> crowd;
	crowd.reserve( static_cast<size_t>( x_count ) * z_count );
	for ( int z = 0; z < z_count; ++z )
	{
		for ( int x = 0; x < x_count; ++x )
		{
			const auto pos = glm::vec3( x - ( x_count / 2 ), 0, z - ( z_count / 2 ) );
			const auto asset_index = gen( rand );
			// The angle is passed as radians: effectively a random angle
			const glm::quat rotation = glm::vec3( 0, gen_angle( rand ), 0 );
			const auto scale = gen_scale( rand );
			crowd.push_back( { asset_index, pos, rotation, scale } );
		}
	}
	return crowd;
}

// One CSV line per character, for reproducing the scene in another engine (glTF conventions: right-handed, Y up)
void dump_crowd( std::span<const crowd_member> crowd, std::span<const char* const> asset_paths, const std::string& path )
{
	std::ofstream out( path );
	std::println( out, "asset,position_x,position_y,position_z,rotation_x,rotation_y,rotation_z,rotation_w,scale" );
	for ( const auto& member : crowd )
	{
		const auto& p = member.position;
		const auto& r = member.rotation;
		std::println( out,
					  "{},{},{},{},{},{},{},{},{}",
					  asset_paths[ member.asset_index ],
					  p.x,
					  p.y,
					  p.z,
					  r.x,
					  r.y,
					  r.z,
					  r.w,
					  member.scale );
	}
}

//#define BISTRO

static constexpr std::array CROWD_ASSETS { "assets/player.glb", "assets/sabrina.glb" };
static constexpr int BENCHMARK_CROWD_SIZE = 100;

int main( int argc, char** argv )
{
	argparse::ArgumentParser command_line( "brutus" );
	command_line.add_argument( "--benchmark" ).flag().help( "Setup scene and camera for the reference benchmark" );
	command_line.add_argument( "--use-vertex-shader" )
		.default_value( false )
		.implicit_value( true )
		.help( "Default to vertex shader rendering even if mesh shaders are available" );
	command_line.add_argument( "--msaa" )
		.default_value( 4 )
		.scan<'i', int>()
		.choices( 1, 2, 4, 8 )
		.help( "MSAA sample count of the main passes, 1 renders without resolves" );
#ifndef BISTRO
	command_line.add_argument( "--dump-scene" ).help( "Dump scene meshes and transforms to the specified file" );
#endif

	try
	{
		command_line.parse_args( argc, argv );
	}
	catch ( const std::exception& err )
	{
		std::cerr << err.what() << std::endl;
		std::cerr << command_line;
		return EINVAL;
	}

	const bool is_benchmark = command_line[ "--benchmark" ] == true;

#ifndef BISTRO
	if ( const auto scene_file = command_line.present( "--dump-scene" ) )
	{
		dump_crowd( generate_crowd( CROWD_ASSETS.size(), BENCHMARK_CROWD_SIZE, BENCHMARK_CROWD_SIZE ), CROWD_ASSETS, *scene_file );
		return 0;
	}
#endif

	constexpr bool profile_startup = true;

	if ( profile_startup )
	{
		PROFILER_START_CAPTURE();
	}

	PROFILER_SCOPE_PUSH( startup, "Startup" );

	tbb_observer observer { };
	observer.observe();

	gpu_stable_power::Context stable_power( is_benchmark );

	PROFILER_SCOPE_PUSH( create_device, "create_device" );
	renderer::Device device( "Brutus-NG [Vulkan]" );
	const auto props = device.get_properties();
	PROFILER_SCOPE_POP( create_device );

	bool vsync = !is_benchmark;
#ifdef __APPLE__
	constexpr auto swapchain_format = renderer::Texture::Format::B8G8R8A8_SRGB;
#else
	constexpr auto swapchain_format = renderer::Texture::Format::R8G8B8A8_SRGB;
#endif

	PROFILER_SCOPE_PUSH( create_swapchain, "create_swapchain" );
	renderer::Swapchain swapchain( device, swapchain_format, vsync );
	PROFILER_SCOPE_POP( create_swapchain );

	{
		PROFILER_SCOPE_N( "init_imgui" );
		ImGui::CreateContext();
		const auto internals = device.get_internals();
		ImGui_ImplSDL3_InitForVulkan( internals.window );
		const VkFormat color_format = static_cast<VkFormat>( swapchain_format );
		ImGui_ImplVulkan_InitInfo info {
			.ApiVersion = internals.api_version,
			.Instance = internals.instance,
			.PhysicalDevice = internals.physical_device,
			.Device = internals.device,
			.QueueFamily = internals.queue_family,
			.Queue = internals.queue,
			.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE,
			.MinImageCount = swapchain.get_image_count(),
			.ImageCount = swapchain.get_image_count(),
			.PipelineInfoMain = { .PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
																   .colorAttachmentCount = 1u,
																   .pColorAttachmentFormats = &color_format } },
			.UseDynamicRendering = true
		};
		ImGui_ImplVulkan_Init( &info );
	}

	BindlessManager bindless_manager( device, std::array { MAX_MESHES, MAX_MATERIALS } );
	renderer::PipelineManager pipelines( device, "shaders", bindless_manager );

	SceneRenderer scene_renderer( device, swapchain_format, command_line.get<int>( "--msaa" ), bindless_manager, pipelines );

	// Bistro needs ~27M
	index_buffer indices( device, 32 * 1024 * 1024 );
	std::vector<asset> assets;
#ifndef BISTRO
	for ( const auto path : CROWD_ASSETS )
	{
		assets.push_back( load_asset( device, bindless_manager, indices, path ) );
	}
#else
	assets.push_back( load_asset( device, bindless_manager, indices, "../niagara_bistro/bistro.gltf" ) );
	//assets.push_back( load_asset( device, bindless_manager, indices, "assets/player.glb" ) );
#endif

	std::array<renderer::raii::TimestampQuery, renderer::MAX_FRAMES_IN_FLIGHT> timestamp_queries { nullptr, nullptr };
	std::array<renderer::raii::StatisticsQuery, renderer::MAX_FRAMES_IN_FLIGHT> statistics_queries { nullptr, nullptr };
	if ( props.statistics_support )
	{
		timestamp_queries = { device.create_timestamp_query( GpuTimings::TIMESTAMPS_COUNT ),
							  device.create_timestamp_query( GpuTimings::TIMESTAMPS_COUNT ) };
		statistics_queries = { device.create_statistics_query(), device.create_statistics_query() };
	};

#ifdef BISTRO
	draw_scene( assets[ 0 ], scene_renderer );
#else
	int x_miku = 0;
	int z_miku = 0;
#endif

	pipelines.wait_ready();

	PROFILER_SCOPE_POP( startup );

	Camera camera { .position = glm::vec3( 0, 1.f, 2.f ) };
	glm::vec3 camera_velocity { };
	GpuTimings gpu_timings( device.get_timestamp_period() );

	if ( profile_startup )
	{
		PROFILER_STOP_CAPTURE();
#ifdef _DEBUG
		PROFILER_SAVE_CAPTURE( "startup_debug.opt" );
#else
		PROFILER_SAVE_CAPTURE( "startup_release.opt" );
#endif
	}

	bool quit = false;
	int frame_count = 0;

	auto time = std::chrono::steady_clock::now();
	constexpr float day_period = 10.f;
	float time_of_day = 0.f;

	bool relative_mouse = false;
	// XXX: we disable occlusion culling support entirely if minmax sampler aren't supported
	// TODO: emulate it in shaders with a gather4 instead
	ui_data ui_state { .vsync = vsync,
					   .stable_power = stable_power.is_enabled(),
					   .mesh_shading = props.mesh_shader_support && command_line[ "--use-vertex-shader" ] == false,
					   .occlusion_culling = props.minmax_filter_support };

	if ( is_benchmark )
	{
		time_of_day = 4.f;
		ui_state.paused = true;
#ifndef BISTRO
		ui_state.x_miku = BENCHMARK_CROWD_SIZE;
		ui_state.z_miku = BENCHMARK_CROWD_SIZE;
		camera.position = { -7.5234f, 2.5384f, 4.8755f };
		camera.pitch = -0.2960f;
		camera.yaw = 0.7039f;
#endif
	}

	while ( !quit )
	{
		PROFILER_FRAME( "MainThread" );
		{
			PROFILER_SCOPE_N( "PollEvents" );
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
					processSDLEvent( event, camera, camera_velocity );
				}
				else
				{
					ImGui_ImplSDL3_ProcessEvent( &event );
				}
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

		camera.position += glm::vec3( camera.calc_rotation_matrix() * glm::vec4( camera_velocity, 0.f ) ) * delta_time;
		// Sun as a point on a sphere: fixed elevation above the horizon, azimuth orbiting over time.
		const float sun_elevation = glm::radians( 60.f ); // height in the sky (0 = horizon, 90 = zenith)
		const float sun_azimuth = time_of_day * std::numbers::pi_v<float> * 2.f / day_period; // full circle over one period
		// Unit vector pointing *toward* the sun. Elevation lives in the angle, so it survives normalize().
		const glm::vec3 to_sun( std::cosf( sun_elevation ) * std::cosf( sun_azimuth ),
								std::sinf( sun_elevation ),
								std::cosf( sun_elevation ) * std::sinf( sun_azimuth ) );
		const glm::vec4 main_light( -to_sun, ui_state.sun_intensity * std::numbers::pi );

		const float sky_illumination = ui_state.sky_to_sun;

		const bool mesh_shading = ui_state.mesh_shading;
		const bool occlusion_culling = ui_state.occlusion_culling;

		auto [ image_index, swapchain_image, swapchain_image_view ] = swapchain.acquire();

		renderer::Statistics stats;
		if ( frame_count > renderer::MAX_FRAMES_IN_FLIGHT )
		{
			std::array<uint64_t, GpuTimings::TIMESTAMPS_COUNT> timestamps { };
			device.get_query_results( timestamp_queries[ image_index ], 0, timestamps );
			gpu_timings.add( timestamps, frame_count );
			stats = device.get_query_results( statistics_queries[ image_index ] );
		}

#ifndef BISTRO
		if ( x_miku != ui_state.x_miku || z_miku != ui_state.z_miku )
		{
			PROFILER_SCOPE_N( "Generate GPU scene" );
			scene_renderer.clear_meshes();
			for ( const auto& member : generate_crowd( assets.size(), ui_state.x_miku, ui_state.z_miku ) )
			{
				const auto& asset = assets[ member.asset_index ];
				const auto scale = glm::vec3( member.scale );
				for ( const auto& mesh : asset.meshes )
				{
					scene_renderer.add_mesh( mesh, member.position, member.rotation, scale, asset.materials[ mesh.material_index ] );
				}
			}
			x_miku = ui_state.x_miku;
			z_miku = ui_state.z_miku;
		}
#endif
		pipelines.update();

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();

		draw_ui( ui_state,
				 props,
				 device.get_memory_usage(),
				 bindless_manager,
				 gpu_timings.averages,
				 stats,
				 scene_renderer.get_statistics(),
				 camera );

		auto command_buffer = device.grab_command_buffer();
		command_buffer->reset();
		command_buffer->begin();

		command_buffer->reset_query( statistics_queries[ image_index ] );
		command_buffer->begin_query( statistics_queries[ image_index ] );

		command_buffer->reset_query( timestamp_queries[ image_index ], 0, GpuTimings::TIMESTAMPS_COUNT );
		GpuTimingScope frame_scope( *command_buffer, timestamp_queries[ image_index ], GpuTiming::FRAME );
		scene_renderer.render( bindless_manager,
							   pipelines,
							   *command_buffer,
							   swapchain_image,
							   swapchain_image_view,
							   mesh_shading ? nullptr : &indices.buffer,
							   timestamp_queries[ image_index ],
							   camera,
							   main_light,
							   sky_illumination,
							   static_cast<float>( ui_state.lod_pixel_tolerance ),
							   ui_state.shadow_distance,
							   static_cast<uint32_t>( ui_state.shadow_cascades ),
							   ui_state.show_shadow_cascades,
							   ui_state.pbr,
							   occlusion_culling );

		{
			PROFILER_GPU_EVENT( "Draw UI" );
			const auto label = command_buffer->debug_label( "UI" );
			const GpuTimingScope ui_scope( *command_buffer, timestamp_queries[ image_index ], GpuTiming::DRAW_UI );
			command_buffer->begin_rendering( { .color_target = { .target = swapchain_image_view } } );
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData( ImGui::GetDrawData(), command_buffer->get_impl() );
			command_buffer->end_rendering();
		}

		command_buffer->texture_barrier( swapchain_image, renderer::ResourceState::COLOR_ATTACHMENT, renderer::ResourceState::PRESENT );
		frame_scope.end();
		command_buffer->end_query( statistics_queries[ image_index ] );

		command_buffer->end();

		swapchain.submit( *command_buffer );

		swapchain.present();
		++frame_count;

		if ( ui_state.vsync != vsync )
		{
			vsync = ui_state.vsync;
			swapchain.recreate( vsync );
			ImGui_ImplVulkan_SetMinImageCount( swapchain.get_image_count() );
		}
		ui_state.stable_power = stable_power.set_enabled( ui_state.stable_power );
	}

	// Wait for the GPU to finish before we roll all the destructors
	device.wait_idle();

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();

	PROFILER_SHUTDOWN();

	return 0;
}
