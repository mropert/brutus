#include "texture_compressor.h"

#include <algorithm>
#include <cassert>
#include <renderer/details/profiler.h>
#include <vector>

namespace
{
	// Matches the push constants of bc_compress.comp
	struct push_constants
	{
		renderer::BindlessSampledTexture source;
		int32_t mip;
		renderer::Extent2D extent;
		uint32_t refinements = 2;
		uint32_t padding = 0;
		renderer::BufferHandle blocks;
	};

	renderer::BindlessTexture make_scratch( renderer::Device& device, BindlessManager& bindless_manager, renderer::Texture::Format format )
	{
		using TexUsage = renderer::Texture::Usage;
		const renderer::Extent2D extent( texture_compressor::SCRATCH_SIZE, texture_compressor::SCRATCH_SIZE );
		return bindless_manager.add_texture(
			device.create_texture( { .format = format,
									 .usage = TexUsage::TRANSFER_SRC | TexUsage::TRANSFER_DST | TexUsage::SAMPLED,
									 .extent = extent,
									 .mips = calc_mip_levels( extent ) } ) );
	}
}

texture_compressor::texture_compressor( renderer::Device& device, BindlessManager& bindless_manager, renderer::PipelineManager& pipelines )
	: _bindless_manager( bindless_manager )
	, _pipelines( pipelines )
	, _scratch_unorm( make_scratch( device, bindless_manager, renderer::Texture::Format::R8G8B8A8_UNORM ) )
	, _scratch_srgb( make_scratch( device, bindless_manager, renderer::Texture::Format::R8G8B8A8_SRGB ) )
{
	using Format = renderer::Texture::Format;
	_variants = { { { Format::BC1_RGB_UNORM, add_pipeline( "1", "0" ) },
					{ Format::BC1_RGB_SRGB, add_pipeline( "1", "1" ) },
					{ Format::BC3_UNORM, add_pipeline( "3", "0" ) },
					{ Format::BC3_SRGB, add_pipeline( "3", "1" ) },
					{ Format::BC5_UNORM, add_pipeline( "5", "0" ) } } };

	// They stay in GENERAL for good: each texture replaces what it uses, and transitioning from UNDEFINED wouldn't wait for
	// the previous texture's dispatches
	auto cmd = device.grab_command_buffer();
	cmd->reset();
	cmd->begin();
	cmd->transition_texture( _scratch_unorm.texture, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL );
	cmd->transition_texture( _scratch_srgb.texture, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL );
	cmd->end();
	device.submit_and_wait( *cmd );
}

void texture_compressor::compress( renderer::CommandBuffer& cmd,
								   const renderer::Buffer& upload,
								   size_t upload_offset,
								   const renderer::Buffer& blocks,
								   size_t blocks_offset,
								   const renderer::Texture& texture ) const
{
	PROFILER_SCOPE();
	const auto format = texture.get_format();
	const bool srgb = format == renderer::Texture::Format::BC1_RGB_SRGB || format == renderer::Texture::Format::BC3_SRGB;
	const auto& scratch = srgb ? _scratch_srgb : _scratch_unorm;
	const renderer::Texture source = scratch.texture;

	// The previous texture's dispatches read the scratch image
	cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::TRANSFER );
	cmd.copy_buffer_to_texture( upload, upload_offset, source, 0, texture.get_extent() );
	for ( int mip = 1; mip < texture.get_mips(); ++mip )
	{
		cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::TRANSFER );
		cmd.blit_texture( source, source, mip - 1, texture.get_extent( mip - 1 ), mip, texture.get_extent( mip ), true );
	}

	cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::COMPUTE );
	const auto pipeline = _pipelines.get( get_pipeline( format ) );
	cmd.bind_pipeline( pipeline, _bindless_manager );
	size_t offset = blocks_offset;
	for ( int mip = 0; mip < texture.get_mips(); ++mip )
	{
		const auto extent = texture.get_extent( mip );
		cmd.push_constants( pipeline,
							push_constants { .source = scratch.handles.sampled_texture,
											 .mip = mip,
											 .extent = extent,
											 .blocks = blocks.get_device_address() + offset } );
		// Each thread compresses a 4x4 block, a workgroup 8x8 blocks
		cmd.dispatch( ( ( extent.width + 3 ) / 4 + 7 ) / 8, ( ( extent.height + 3 ) / 4 + 7 ) / 8, 1 );
		offset += texture.get_mip_size( mip );
	}

	cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::TRANSFER );
	cmd.transition_texture( texture, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL );
	offset = blocks_offset;
	for ( int mip = 0; mip < texture.get_mips(); ++mip )
	{
		cmd.copy_buffer_to_texture( blocks, offset, texture, mip );
		offset += texture.get_mip_size( mip );
	}
}

renderer::PipelineHandle texture_compressor::add_pipeline( const char* bc_format, const char* srgb )
{
	const std::vector<renderer::ShaderSource::Define> defines { { "BC_FORMAT", bc_format }, { "SRGB", srgb } };
	return _pipelines.add( renderer::Pipeline::Desc { .push_constants_size = sizeof( push_constants ) },
						   { { .path = "bc_compress.comp", .stage = renderer::ShaderStage::COMPUTE, .defines = defines } } );
}

renderer::PipelineHandle texture_compressor::get_pipeline( renderer::Texture::Format format ) const
{
	const auto it = std::find_if( begin( _variants ), end( _variants ), [ format ]( const auto& v ) { return v.first == format; } );
	assert( it != end( _variants ) );
	return it->second;
}
