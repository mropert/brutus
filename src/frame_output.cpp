#include "frame_output.h"

#include "gpu_timing.h"

#include <SDL3_image/SDL_image.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <renderer/command_buffer.h>
#include <renderer/details/profiler.h>

WindowOutput::WindowOutput( renderer::Device& device, renderer::Texture::Format format, bool vsync )
	: _swapchain(
		  [ & ]
		  {
			  PROFILER_SCOPE_N( "create_swapchain" );
			  return renderer::Swapchain( device, format, vsync );
		  }() )
{
	PROFILER_SCOPE_N( "init_imgui" );
	const auto internals = device.get_internals();
	ImGui_ImplSDL3_InitForVulkan( internals.window );
	const VkFormat color_format = static_cast<VkFormat>( format );
	ImGui_ImplVulkan_InitInfo info {
		.ApiVersion = internals.api_version,
		.Instance = internals.instance,
		.PhysicalDevice = internals.physical_device,
		.Device = internals.device,
		.QueueFamily = internals.queue_family,
		.Queue = internals.queue,
		.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE,
		.MinImageCount = _swapchain.get_image_count(),
		.ImageCount = _swapchain.get_image_count(),
		.PipelineInfoMain = { .PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
															   .colorAttachmentCount = 1u,
															   .pColorAttachmentFormats = &color_format } },
		.UseDynamicRendering = true
	};
	ImGui_ImplVulkan_Init( &info );
}

WindowOutput::~WindowOutput()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplSDL3_Shutdown();
}

void WindowOutput::process_event( const SDL_Event& event )
{
	ImGui_ImplSDL3_ProcessEvent( &event );
}

FrameOutput::Frame WindowOutput::begin_frame()
{
	auto [ frame_index, swapchain_image, swapchain_image_view ] = _swapchain.acquire();
	_current = { frame_index, swapchain_image, swapchain_image_view };

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL3_NewFrame();

	return _current;
}

void WindowOutput::end_frame( renderer::CommandBuffer& command_buffer, renderer::TimestampQuery timestamps )
{
	{
		PROFILER_GPU_EVENT( "Draw UI" );
		const auto label = command_buffer.debug_label( "UI" );
		const GpuTimingScope ui_scope( command_buffer, timestamps, GpuTiming::DRAW_UI );
		command_buffer.begin_rendering( { .color_target = { .target = _current.target_view } } );
		ImGui_ImplVulkan_RenderDrawData( ImGui::GetDrawData(), command_buffer.get_impl() );
		command_buffer.end_rendering();
	}

	command_buffer.transition_texture( _current.target, renderer::Texture::Layout::GENERAL, renderer::Texture::Layout::PRESENT_SRC );
}

void WindowOutput::submit_and_present( renderer::CommandBuffer& command_buffer )
{
	command_buffer.end();
	_swapchain.submit( command_buffer );
	_swapchain.present();
}

void WindowOutput::set_vsync( bool vsync )
{
	_swapchain.recreate( vsync );
	ImGui_ImplVulkan_SetMinImageCount( _swapchain.get_image_count() );
}

CaptureOutput::CaptureOutput( renderer::Device& device, renderer::Texture::Format format )
	: _device( device )
	, _fence( _device.create_fence( true ) )
	, _texture( _device.create_texture( { .format = format,
										  .usage = renderer::Texture::Usage::TRANSFER_SRC | renderer::Texture::Usage::TRANSFER_DST
											  | renderer::Texture::Usage::COLOR_ATTACHMENT,
										  .extent = _device.get_extent() } ) )
	, _view( _device.create_texture_view( _texture, renderer::TextureView::Aspect::COLOR ) )
	, _buffer( _device.create_buffer( renderer::Buffer::Usage::TRANSFER_DST,
									  _texture.get_mip_size( 0 ),
									  renderer::Buffer::CpuUsage::READ_WRITE ) )
{
	// No renderer backend: ImGui still builds draw data, which is never rendered. Textures it asks for are never created.
	auto& io = ImGui::GetIO();
	io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
	io.DisplaySize = ImVec2( static_cast<float>( _device.get_extent().width ), static_cast<float>( _device.get_extent().height ) );
}

FrameOutput::Frame CaptureOutput::begin_frame()
{
	_device.wait_for_fences( { *_fence }, UINT64_MAX );
	_device.reset_fences( { *_fence } );

	// No platform backend to measure it
	ImGui::GetIO().DeltaTime = 1.f / 60.f;

	return { _frame_count++ % renderer::MAX_FRAMES_IN_FLIGHT, _texture, _view };
}

void CaptureOutput::submit_and_present( renderer::CommandBuffer& command_buffer )
{
	command_buffer.end();
	_device.submit( command_buffer, *_fence );
}

void CaptureOutput::save_to_file( const std::string& path )
{
	_device.wait_for_fences( { *_fence }, UINT64_MAX );
	{
		auto cmd = _device.grab_command_buffer();
		cmd->reset();
		cmd->begin();
		// Makes no assumption on how the frame was composited: draw, storage write or blit
		cmd->barrier( renderer::Stage::ANY_SHADER | renderer::Stage::COLOR | renderer::Stage::TRANSFER, renderer::Stage::TRANSFER );
		cmd->copy_texture_to_buffer( _texture, 0, _buffer );
		cmd->barrier( renderer::Stage::TRANSFER, renderer::Stage::HOST );
		cmd->end();
		_device.submit_and_wait( *cmd );
	}
	_buffer.flush();

	SDL_PixelFormat pixel_format;
	switch ( _texture.get_format() )
	{
		case renderer::Texture::Format::R8G8B8A8_UNORM:
		case renderer::Texture::Format::R8G8B8A8_SRGB:
			pixel_format = SDL_PIXELFORMAT_RGBA32;
			break;
		case renderer::Texture::Format::B8G8R8A8_UNORM:
		case renderer::Texture::Format::B8G8R8A8_SRGB:
			pixel_format = SDL_PIXELFORMAT_BGRA32;
			break;
		default:
			throw renderer::Error( "Unsupported capture format" );
	}

	const auto extent = _texture.get_extent();
	sdl::raii::Surface surface( SDL_CreateSurfaceFrom( static_cast<int>( extent.width ),
													   static_cast<int>( extent.height ),
													   pixel_format,
													   _buffer.get_mapped_address(),
													   static_cast<int>( extent.width * 4 ) ) );
	if ( !surface )
	{
		throw renderer::Error( SDL_GetError() );
	}
	if ( !IMG_SavePNG( surface.get(), path.c_str() ) )
	{
		throw renderer::Error( SDL_GetError() );
	}
}
