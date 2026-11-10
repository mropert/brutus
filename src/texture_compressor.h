#pragma once

#include "asset.h"

#include <array>
#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/command_buffer.h>
#include <renderer/device.h>
#include <renderer/pipeline_manager.h>
#include <renderer/texture.h>
#include <utility>

// Compresses decoded textures to BC formats on the GPU, see docs/texture_compression.md. Mips are generated on an
// uncompressed scratch image, then each one is compressed to a buffer and copied to the final texture.
class texture_compressor
{
public:
	// Upper bound of the textures that can be compressed
	static constexpr uint32_t SCRATCH_SIZE = 2048;

	texture_compressor( renderer::Device& device, BindlessManager& bindless_manager, renderer::PipelineManager& pipelines );

	// Must be called before compressing: pipelines can't be fetched while others are being added
	void wait_ready() { _pipelines.wait_ready(); }
	bool is_ready() { return _pipelines.is_ready(); }

	// Records the compression of a tightly packed RGBA8 image at upload_offset to texture, which has a full mip chain.
	// blocks must have room for all its mips at blocks_offset.
	void compress( renderer::CommandBuffer& cmd,
				   const renderer::Buffer& upload,
				   size_t upload_offset,
				   const renderer::Buffer& blocks,
				   size_t blocks_offset,
				   const renderer::Texture& texture ) const;

private:
	renderer::PipelineHandle add_pipeline( const char* bc_format, const char* srgb );
	renderer::PipelineHandle get_pipeline( renderer::Texture::Format format ) const;

	const BindlessManager& _bindless_manager;
	renderer::PipelineManager& _pipelines;
	renderer::BindlessTexture _scratch_unorm;
	renderer::BindlessTexture _scratch_srgb;
	std::array<std::pair<renderer::Texture::Format, renderer::PipelineHandle>, 5> _variants;
};
