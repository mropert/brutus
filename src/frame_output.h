#pragma once

#include <SDL3/SDL_events.h>
#include <renderer/buffer.h>
#include <renderer/device.h>
#include <renderer/swapchain.h>
#include <renderer/texture.h>
#include <string>

// Where frames go: a window's swapchain with the UI drawn on top, or an offscreen target that can be saved to a file.
// The ImGui context is owned by the caller, outputs only set up the backends they need.
class FrameOutput
{
public:
	struct Frame
	{
		uint32_t index; // Frame in flight, in [0, MAX_FRAMES_IN_FLIGHT)
		renderer::Texture target;
		renderer::TextureView target_view;
	};

	virtual ~FrameOutput() = default;
	virtual void process_event( const SDL_Event& event ) = 0;
	// Must be called before ImGui::NewFrame()
	virtual Frame begin_frame() = 0;
	// Records the UI and the transitions the output needs, the command buffer stays open
	virtual void end_frame( renderer::CommandBuffer& command_buffer, renderer::TimestampQuery timestamps ) = 0;
	// Ends the command buffer and submits it
	virtual void submit_and_present( renderer::CommandBuffer& command_buffer ) = 0;
	// Must be called between frames
	virtual void set_vsync( bool vsync ) = 0;
};

class WindowOutput final : public FrameOutput
{
public:
	WindowOutput( renderer::Device& device, renderer::Texture::Format format, bool vsync );
	~WindowOutput() override;
	void process_event( const SDL_Event& event ) override;
	Frame begin_frame() override;
	void end_frame( renderer::CommandBuffer& command_buffer, renderer::TimestampQuery timestamps ) override;
	void submit_and_present( renderer::CommandBuffer& command_buffer ) override;
	void set_vsync( bool vsync ) override;

private:
	renderer::Swapchain _swapchain;
	Frame _current;
};

// Renders without UI, one frame at a time
class CaptureOutput final : public FrameOutput
{
public:
	CaptureOutput( renderer::Device& device, renderer::Texture::Format format );
	void process_event( const SDL_Event& ) override { }
	Frame begin_frame() override;
	void end_frame( renderer::CommandBuffer&, renderer::TimestampQuery ) override { }
	void submit_and_present( renderer::CommandBuffer& command_buffer ) override;
	void set_vsync( bool ) override { }
	// Saves the last submitted frame as a PNG, must be called between frames
	void save_to_file( const std::string& path );

private:
	renderer::Device& _device;
	renderer::raii::Fence _fence;
	renderer::raii::Texture _texture;
	renderer::raii::TextureView _view;
	renderer::raii::Buffer _buffer;
	uint32_t _frame_count = 0;
};
