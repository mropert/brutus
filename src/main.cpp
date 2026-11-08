#include <SDL3/SDL.h>
#include <argparse/argparse.hpp>
#include <array>
#include <chrono>
#include <format>
#include <fstream>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_aligned.hpp>
#include <imgui.h>
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
#include <renderer/texture.h>

//

#include "asset.h"
#include "camera.h"
#include "frame_output.h"
#include "gpu_timing.h"
#include "scene_renderer.h"

static constexpr uint32_t MAX_MESHES = 4096;
static constexpr uint32_t MAX_MATERIALS = 8192;
static constexpr float MIN_SHADOW_DISTANCE = 1.f;
static constexpr float MAX_SHADOW_DISTANCE = 100.f;

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
	bool crowd; // The crowd scene, its size can be changed
	bool grid; // The ground plane, only useful in scenes without their own ground
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
		ImGui::SliderFloat( "Shadow distance",
							&data.shadow_distance,
							MIN_SHADOW_DISTANCE,
							MAX_SHADOW_DISTANCE,
							"%.3f",
							ImGuiSliderFlags_AlwaysClamp );
		ImGui::SliderInt( "Shadow cascades", &data.shadow_cascades, 0, SceneRenderer::SHADOW_CASCADE_MAX );
		ImGui::Checkbox( "Show shadow cascades", &data.show_shadow_cascades );
		ImGui::Checkbox( "Ground grid", &data.grid );
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
		if ( data.crowd )
		{
			ImGui::Text( "Instance count" );
			ImGui::SliderInt( "x", &data.x_miku, 1, 100 );
			ImGui::SliderInt( "z", &data.z_miku, 1, 100 );
		}
		ImGui::SliderInt( "LOD pixel tolerance", &data.lod_pixel_tolerance, 1, 8 );
	}
	ImGui::End();
}

void draw_scene( const Assets& assets, const scene& scene, SceneRenderer& scene_renderer )
{
	scene_renderer.clear_meshes();
	for ( const auto& instance : scene.instances )
	{
		const auto& mesh = assets.get_mesh( instance.mesh );
		scene_renderer.add_mesh( mesh, instance.translation, instance.rotation, instance.scale, assets.get_material( mesh.material ) );
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

// Places the models' instances at each member's transform
scene make_crowd_scene( std::span<const scene> models, std::span<const crowd_member> crowd )
{
	scene crowd_scene;
	for ( const auto& member : crowd )
	{
		for ( const auto& instance : models[ member.asset_index ].instances )
		{
			// Exact composition of the two transforms: the member's scale is uniform
			crowd_scene.instances.push_back( { .translation = member.position + member.rotation * ( member.scale * instance.translation ),
											   .rotation = member.rotation * instance.rotation,
											   .scale = member.scale * instance.scale,
											   .mesh = instance.mesh } );
		}
	}
	return crowd_scene;
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

static constexpr std::array CROWD_ASSETS { "assets/player.glb", "assets/sabrina.glb" };
static constexpr int BENCHMARK_CROWD_SIZE = 100;

auto store_bool( bool& value )
{
	using namespace std::literals;
	return [ & ]( std::string_view v )
	{
		if ( v == "1"sv || v == "true"sv || v == "on"sv )
		{
			value = true;
		}
		else if ( v == "0"sv || v == "false"sv || v == "off"sv )
		{
			value = false;
		}
		else
		{
			throw std::runtime_error( "Invalid value for boolean argument, must be one of 1, 0, true, false, on or off" );
		}
	};
}

int main( int argc, char** argv )
{
	struct initial_settings
	{
		int msaa = 4;
		int shadow_cascades = 4;
		int shadow_distance = 25;
		bool benchmark = false;
		bool mesh_shading = true;
		bool occlusion_culling = true;
		bool show_cascades = false;
		int capture_frame = 16;
		std::string texture_compression = "none";
		bool no_dds = false;
	};

	initial_settings settings;

	argparse::ArgumentParser command_line( "brutus" );
	command_line.add_argument( "--benchmark" )
		.store_into( settings.benchmark )
		.help( "Setup scene and camera for the reference benchmark" );
	command_line.add_argument( "--capture" ).help( "Render without window or UI, save the last frame to the specified PNG and quit" );
	command_line.add_argument( "--capture-frame" )
		.scan<'i', int>()
		.store_into( settings.capture_frame )
		.help( "Frame saved by --capture, counting from 1: earlier frames warm up temporal state" );
	command_line.add_argument( "--dump-scene" ).help( "Dump the crowd's meshes and transforms to the specified file" );
	command_line.add_argument( "--no-dds" )
		.store_into( settings.no_dds )
		.help( "Ignore DDS images in glTF scenes when there's another image" );
	command_line.add_argument( "--texture-compression" )
		.choices( "none", "gpu" )
		.store_into( settings.texture_compression )
		.help( "How images that aren't already block compressed are stored" );
	command_line.add_argument( "--mesh-shader" ).action( store_bool( settings.mesh_shading ) ).help( "Use mesh shaders if supported" );
	command_line.add_argument( "--msaa" )
		.scan<'i', int>()
		.choices( 1, 2, 4, 8 )
		.store_into( settings.msaa )
		.help( "MSAA sample count of the main passes, 1 renders without resolves" );
	command_line.add_argument( "--occlusion-culling" ).action( store_bool( settings.occlusion_culling ) ).help( "Use occlusion culling" );
	command_line.add_argument( "--scene" ).help( "glTF scene to render instead of the crowd" );
	command_line.add_argument( "--shadow-cascades" )
		.scan<'i', int>()
		.choices( 0, 1, 2, 3, 4 )
		.store_into( settings.shadow_cascades )
		.help( "Shadow cascades count, or 0 for no shadows" );
	command_line.add_argument( "--shadow-distance" )
		.scan<'i', int>()
		.store_into( settings.shadow_distance )
		.help( "Maximum shadow distance" );
	command_line.add_argument( "--show-cascades" )
		.action( store_bool( settings.show_cascades ) )
		.help( "Show shadow cascades as a tinted filter" );

	try
	{
		command_line.parse_args( argc, argv );
	}
	catch ( const std::exception& err )
	{
		std::print( stderr, "{}\n", err.what() );
		std::cerr << command_line;
		return EINVAL;
	}

	// Extra validation that can't be done within the parser
	if ( settings.shadow_distance < MIN_SHADOW_DISTANCE || settings.shadow_distance > MAX_SHADOW_DISTANCE )
	{
		std::print( stderr, "Shadow distance must be between {} and {}\n", MIN_SHADOW_DISTANCE, MAX_SHADOW_DISTANCE );
		return EINVAL;
	}

	if ( settings.capture_frame < 1 )
	{
		std::print( stderr, "Capture frame must be at least 1\n" );
		return EINVAL;
	}

	const auto scene_path = command_line.present( "--scene" );
	const auto capture_path = command_line.present( "--capture" );
	if ( command_line.is_used( "--capture-frame" ) && !capture_path )
	{
		std::print( stderr, "--capture-frame requires --capture\n" );
		return EINVAL;
	}

	if ( const auto scene_file = command_line.present( "--dump-scene" ) )
	{
		if ( scene_path )
		{
			std::cerr << "--dump-scene only dumps the crowd, it can't be combined with --scene" << std::endl;
			return EINVAL;
		}
		dump_crowd( generate_crowd( CROWD_ASSETS.size(), BENCHMARK_CROWD_SIZE, BENCHMARK_CROWD_SIZE ), CROWD_ASSETS, *scene_file );
		return 0;
	}

	constexpr bool profile_startup = true;

	if ( profile_startup )
	{
		PROFILER_START_CAPTURE();
	}

	PROFILER_SCOPE_PUSH( startup, "Startup" );

	tbb_observer observer { };
	observer.observe();

	gpu_stable_power::Context stable_power( settings.benchmark );

	PROFILER_SCOPE_PUSH( create_device, "create_device" );
	renderer::Device device( "Brutus-NG [Vulkan]",
							 capture_path ? renderer::Device::WindowMode::HIDDEN : renderer::Device::WindowMode::BORDERLESS );
	const auto props = device.get_properties();
	PROFILER_SCOPE_POP( create_device );

	bool vsync = !settings.benchmark;
#ifdef __APPLE__
	constexpr auto swapchain_format = renderer::Texture::Format::B8G8R8A8_SRGB;
#else
	constexpr auto swapchain_format = renderer::Texture::Format::R8G8B8A8_SRGB;
#endif

	// Built on both outputs, only the window renders it
	ImGui::CreateContext();
	std::unique_ptr<FrameOutput> output;
	CaptureOutput* capture_output = nullptr;
	if ( capture_path )
	{
		auto capture = std::make_unique<CaptureOutput>( device, swapchain_format );
		capture_output = capture.get();
		output = std::move( capture );
	}
	else
	{
		output = std::make_unique<WindowOutput>( device, swapchain_format, vsync );
	}

	BindlessManager bindless_manager( device, std::array { MAX_MESHES, MAX_MATERIALS } );
	renderer::PipelineManager pipelines( device, "shaders", bindless_manager );

	SceneRenderer scene_renderer( device, swapchain_format, settings.msaa, bindless_manager, pipelines );

	// Bistro needs ~27M
	Assets assets( device,
				   bindless_manager,
				   pipelines,
				   32 * 1024 * 1024,
				   { .compression = settings.texture_compression == "gpu" ? texture_compression::GPU : texture_compression::NONE,
					 .use_dds = !settings.no_dds } );
	std::vector<scene> crowd_models;
	scene gltf_scene;
	if ( scene_path )
	{
		gltf_scene = assets.load( *scene_path );
		draw_scene( assets, gltf_scene, scene_renderer );
	}
	else
	{
		for ( const auto path : CROWD_ASSETS )
		{
			crowd_models.push_back( assets.load( path ) );
		}
	}

	std::array<renderer::raii::TimestampQuery, renderer::MAX_FRAMES_IN_FLIGHT> timestamp_queries { nullptr, nullptr };
	std::array<renderer::raii::StatisticsQuery, renderer::MAX_FRAMES_IN_FLIGHT> statistics_queries { nullptr, nullptr };
	if ( props.statistics_support )
	{
		timestamp_queries = { device.create_timestamp_query( GpuTimings::TIMESTAMPS_COUNT ),
							  device.create_timestamp_query( GpuTimings::TIMESTAMPS_COUNT ) };
		statistics_queries = { device.create_statistics_query(), device.create_statistics_query() };
	};

	int x_miku = 0;
	int z_miku = 0;

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
	// The sun orbits around the vertical axis at a fixed elevation (0 = horizon, pi / 2 = zenith), starting from the
	// scene's sun if it has one
	float sun_elevation = glm::radians( 60.f );
	float sun_start_azimuth = 0.f;
	if ( gltf_scene.sun_direction )
	{
		const auto to_sun = -*gltf_scene.sun_direction;
		sun_elevation = std::asin( to_sun.y );
		sun_start_azimuth = std::atan2( to_sun.z, to_sun.x );
	}

	bool relative_mouse = false;
	// XXX: we disable occlusion culling support entirely if minmax sampler aren't supported
	// TODO: emulate it in shaders with a gather4 instead
	ui_data ui_state { .shadow_distance = static_cast<float>( settings.shadow_distance ),
					   .shadow_cascades = settings.shadow_cascades,
					   .show_shadow_cascades = settings.show_cascades,
					   .crowd = !scene_path,
					   .grid = !scene_path,
					   .vsync = vsync,
					   .stable_power = stable_power.is_enabled(),
					   .mesh_shading = props.mesh_shader_support && settings.mesh_shading,
					   .occlusion_culling = props.minmax_filter_support && settings.occlusion_culling };

	if ( !ui_state.crowd )
	{
		// Start with the scene's own lighting and view
		ui_state.paused = true;
		if ( gltf_scene.camera )
		{
			camera = *gltf_scene.camera;
		}
	}
	else if ( settings.benchmark )
	{
		time_of_day = 4.f;
		ui_state.paused = true;
		ui_state.x_miku = BENCHMARK_CROWD_SIZE;
		ui_state.z_miku = BENCHMARK_CROWD_SIZE;
		camera.position = { -7.5234f, 2.5384f, 4.8755f };
		camera.pitch = -0.2960f;
		camera.yaw = 0.7039f;
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
					output->process_event( event );
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
		const float sun_azimuth = sun_start_azimuth
			+ time_of_day * std::numbers::pi_v<float> * 2.f / day_period; // full circle over one period
		// Unit vector pointing *toward* the sun. Elevation lives in the angle, so it survives normalize().
		const glm::vec3 to_sun( std::cosf( sun_elevation ) * std::cosf( sun_azimuth ),
								std::sinf( sun_elevation ),
								std::cosf( sun_elevation ) * std::sinf( sun_azimuth ) );
		const glm::vec4 main_light( -to_sun, ui_state.sun_intensity * std::numbers::pi );

		const float sky_illumination = ui_state.sky_to_sun;

		const bool mesh_shading = ui_state.mesh_shading;
		const bool occlusion_culling = ui_state.occlusion_culling;

		const auto [ image_index, output_image, output_image_view ] = output->begin_frame();

		renderer::Statistics stats;
		if ( frame_count > renderer::MAX_FRAMES_IN_FLIGHT )
		{
			std::array<uint64_t, GpuTimings::TIMESTAMPS_COUNT> timestamps { };
			device.get_query_results( timestamp_queries[ image_index ], 0, timestamps );
			gpu_timings.add( timestamps, frame_count );
			stats = device.get_query_results( statistics_queries[ image_index ] );
		}

		if ( ui_state.crowd && ( x_miku != ui_state.x_miku || z_miku != ui_state.z_miku ) )
		{
			PROFILER_SCOPE_N( "Generate GPU scene" );
			draw_scene( assets,
						make_crowd_scene( crowd_models, generate_crowd( crowd_models.size(), ui_state.x_miku, ui_state.z_miku ) ),
						scene_renderer );
			x_miku = ui_state.x_miku;
			z_miku = ui_state.z_miku;
		}
		pipelines.update();

		ImGui::NewFrame();

		draw_ui( ui_state,
				 props,
				 device.get_memory_usage(),
				 bindless_manager,
				 gpu_timings.averages,
				 stats,
				 scene_renderer.get_statistics(),
				 camera );

		ImGui::Render();

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
							   output_image,
							   output_image_view,
							   mesh_shading ? nullptr : &assets.get_indices().buffer,
							   timestamp_queries[ image_index ],
							   camera,
							   main_light,
							   sky_illumination,
							   static_cast<float>( ui_state.lod_pixel_tolerance ),
							   ui_state.shadow_distance,
							   static_cast<uint32_t>( ui_state.shadow_cascades ),
							   ui_state.grid,
							   ui_state.show_shadow_cascades,
							   ui_state.pbr,
							   occlusion_culling );

		output->end_frame( *command_buffer, timestamp_queries[ image_index ] );
		frame_scope.end();
		command_buffer->end_query( statistics_queries[ image_index ] );

		output->submit_and_present( *command_buffer );
		++frame_count;

		if ( capture_output && frame_count == settings.capture_frame )
		{
			capture_output->save_to_file( *capture_path );
			quit = true;
		}

		if ( ui_state.vsync != vsync )
		{
			vsync = ui_state.vsync;
			output->set_vsync( vsync );
		}
		ui_state.stable_power = stable_power.set_enabled( ui_state.stable_power );
	}

	// Wait for the GPU to finish before we roll all the destructors
	device.wait_idle();

	output.reset();
	ImGui::DestroyContext();

	PROFILER_SHUTDOWN();

	return 0;
}
