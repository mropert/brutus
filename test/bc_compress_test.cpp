// Runs shaders/bc_compress.comp on images and measures the PSNR of each mip, decoded by the GPU: the blocks are copied to
// a BC texture, which is blitted to RGBA8 and read back.
//
// Usage, from the repository root: bc-compress-test <output directory> <image>...

#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <print>
#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/command_buffer.h>
#include <renderer/details/profiler.h>
#include <renderer/device.h>
#include <renderer/pipeline_manager.h>
#include <renderer/shader.h>
#include <renderer/texture.h>
#include <string>
#include <vector>

namespace
{
	using Bindless = renderer::BindlessManager<uint32_t>;
	using Format = renderer::Texture::Format;
	using TexUsage = renderer::Texture::Usage;
	using BufUsage = renderer::Buffer::Usage;

	// Matches the push constants of bc_compress.comp
	struct CompressConstants
	{
		uint32_t source;
		int32_t mip;
		int32_t width;
		int32_t height;
		uint32_t refinements;
		uint32_t padding = 0;
		renderer::BufferHandle blocks;
	};

	struct Mode
	{
		const char* name;
		const char* bc_format;
		const char* srgb;
		Format compressed;
		Format source;
		int channels;
		bool uses_refinements;
		renderer::PipelineHandle pipeline = { };
	};

	struct Source
	{
		renderer::BindlessTexture texture;
		renderer::raii::Buffer readback;
	};

	constexpr int RGBA_SIZE = 4;

	template <typename F>
	void record_and_wait( renderer::Device& device, F&& record )
	{
		auto cmd = device.grab_command_buffer();
		cmd->reset();
		cmd->begin();
		record( *cmd );
		cmd->end();
		device.submit_and_wait( *cmd );
	}

	std::size_t total_size( const renderer::Texture& texture )
	{
		std::size_t size = 0;
		for ( int mip = 0; mip < texture.get_mips(); ++mip )
		{
			size += texture.get_mip_size( mip );
		}
		return size;
	}

	renderer::raii::Buffer make_readback( renderer::Device& device, std::size_t size )
	{
		return device.create_buffer( BufUsage::TRANSFER_DST, size, renderer::Buffer::CpuUsage::READ_WRITE );
	}

	void copy_mips_to_buffer( renderer::CommandBuffer& cmd, const renderer::Texture& texture, const renderer::Buffer& buffer )
	{
		std::size_t offset = 0;
		for ( int mip = 0; mip < texture.get_mips(); ++mip )
		{
			cmd.copy_texture_to_buffer( texture, mip, buffer, offset );
			offset += texture.get_mip_size( mip );
		}
	}

	// Uploads tightly packed RGBA8 data to mip 0, generates the other mips and reads them all back
	Source
	make_source( renderer::Device& device, Bindless& bindless, const void* pixels, renderer::Extent2D extent, Format format, int mips )
	{
		auto texture = device.create_texture( { .format = format,
												.usage = TexUsage::TRANSFER_SRC | TexUsage::TRANSFER_DST | TexUsage::SAMPLED,
												.extent = extent,
												.mips = mips } );
		const auto mip0_size = texture.get_mip_size( 0 );
		auto staging = device.create_buffer( BufUsage::TRANSFER_SRC, mip0_size, renderer::Buffer::CpuUsage::WRITE );
		std::memcpy( staging.get_mapped_address(), pixels, mip0_size );

		Source source { .texture = bindless.add_texture( std::move( texture ) ) };
		const renderer::Texture tex = source.texture.texture;
		source.readback = make_readback( device, total_size( tex ) );

		record_and_wait( device,
						 [ & ]( renderer::CommandBuffer& cmd )
						 {
							 cmd.transition_texture( tex, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL );
							 cmd.copy_buffer_to_texture( staging, 0, tex );
							 for ( int mip = 1; mip < tex.get_mips(); ++mip )
							 {
								 cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::TRANSFER );
								 cmd.blit_texture( tex, tex, mip - 1, mip, true );
							 }
							 cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::TRANSFER );
							 copy_mips_to_buffer( cmd, tex, source.readback );
							 cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::COMPUTE | renderer::Stage::HOST );
						 } );
		source.readback.flush();
		return source;
	}

	renderer::raii::Buffer make_block_buffer( renderer::Device& device, std::size_t size )
	{
		return device.create_buffer( BufUsage::STORAGE_BUFFER | BufUsage::SHADER_DEVICE_ADDRESS | BufUsage::TRANSFER_SRC, size );
	}

	void dispatch_mip( renderer::CommandBuffer& cmd,
					   const renderer::Pipeline& pipeline,
					   const renderer::BindlessTexture& source,
					   const renderer::Buffer& blocks,
					   std::size_t offset,
					   int mip,
					   uint32_t refinements )
	{
		const auto extent = source.texture.get_extent( mip );
		cmd.push_constants( pipeline,
							CompressConstants { .source = source.handles.sampled_texture.index,
												.mip = mip,
												.width = static_cast<int32_t>( extent.width ),
												.height = static_cast<int32_t>( extent.height ),
												.refinements = refinements,
												.blocks = blocks.get_device_address() + offset } );
		// 8x8 blocks per workgroup
		const uint32_t blocks_x = ( extent.width + 3 ) / 4;
		const uint32_t blocks_y = ( extent.height + 3 ) / 4;
		cmd.dispatch( ( blocks_x + 7 ) / 8, ( blocks_y + 7 ) / 8, 1 );
	}

	void save_png( const std::filesystem::path& path, void* pixels, renderer::Extent2D extent )
	{
		const sdl::raii::Surface surface( SDL_CreateSurfaceFrom( static_cast<int>( extent.width ),
																 static_cast<int>( extent.height ),
																 SDL_PIXELFORMAT_RGBA32,
																 pixels,
																 static_cast<int>( extent.width ) * RGBA_SIZE ) );
		if ( !surface || !IMG_SavePNG( surface.get(), path.string().c_str() ) )
		{
			throw renderer::Error( SDL_GetError() );
		}
	}

	double to_psnr( double squared_error, std::size_t samples )
	{
		if ( squared_error == 0.0 )
		{
			return std::numeric_limits<double>::infinity();
		}
		return 10.0 * std::log10( 255.0 * 255.0 * static_cast<double>( samples ) / squared_error );
	}

	void compress_and_measure( renderer::Device& device,
							   Bindless& bindless,
							   const renderer::PipelineManager& pipelines,
							   const Mode& mode,
							   const Source& source,
							   uint32_t refinements,
							   const std::string& image_name,
							   const std::filesystem::path& out_dir )
	{
		const renderer::Texture src = source.texture.texture;
		const auto extent = src.get_extent();
		auto compressed = device.create_texture( { .format = mode.compressed,
												   .usage = TexUsage::TRANSFER_SRC | TexUsage::TRANSFER_DST,
												   .extent = extent,
												   .mips = src.get_mips() } );
		auto decoded = device.create_texture(
			{ .format = mode.source, .usage = TexUsage::TRANSFER_SRC | TexUsage::TRANSFER_DST, .extent = extent, .mips = src.get_mips() } );
		auto blocks = make_block_buffer( device, total_size( compressed ) );
		auto readback = make_readback( device, total_size( decoded ) );
		const auto& pipeline = pipelines.get( mode.pipeline );

		const auto start = std::chrono::steady_clock::now();
		record_and_wait( device,
						 [ & ]( renderer::CommandBuffer& cmd )
						 {
							 cmd.bind_pipeline( pipeline, bindless );
							 std::size_t offset = 0;
							 for ( int mip = 0; mip < src.get_mips(); ++mip )
							 {
								 dispatch_mip( cmd, pipeline, source.texture, blocks, offset, mip, refinements );
								 offset += compressed.get_mip_size( mip );
							 }
						 } );
		const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;

		record_and_wait( device,
						 [ & ]( renderer::CommandBuffer& cmd )
						 {
							 cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::TRANSFER );
							 cmd.transition_texture( compressed, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL );
							 cmd.transition_texture( decoded, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL );
							 std::size_t offset = 0;
							 for ( int mip = 0; mip < src.get_mips(); ++mip )
							 {
								 cmd.copy_buffer_to_texture( blocks, offset, compressed, mip );
								 offset += compressed.get_mip_size( mip );
							 }
							 cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::TRANSFER );
							 for ( int mip = 0; mip < src.get_mips(); ++mip )
							 {
								 cmd.blit_texture( compressed, decoded, mip, mip );
							 }
							 cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::TRANSFER );
							 copy_mips_to_buffer( cmd, decoded, readback );
							 cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::HOST );
						 } );
		readback.flush();

		const auto expected = static_cast<const uint8_t*>( source.readback.get_mapped_address() );
		const auto actual = static_cast<const uint8_t*>( readback.get_mapped_address() );
		std::size_t offset = 0;
		double chain_error = 0.0;
		std::size_t chain_samples = 0;
		double mip0_psnr = 0.0;
		double worst_psnr = std::numeric_limits<double>::infinity();
		int worst_mip = 0;
		std::size_t worst_offset = 0;
		for ( int mip = 0; mip < src.get_mips(); ++mip )
		{
			const auto size = src.get_mip_size( mip );
			double error = 0.0;
			for ( std::size_t i = 0; i < size; i += RGBA_SIZE )
			{
				for ( std::size_t channel = 0; channel < static_cast<std::size_t>( mode.channels ); ++channel )
				{
					const double delta = static_cast<double>( expected[ offset + i + channel ] )
						- static_cast<double>( actual[ offset + i + channel ] );
					error += delta * delta;
				}
			}
			const auto samples = size / RGBA_SIZE * mode.channels;
			const auto psnr = to_psnr( error, samples );
			mip0_psnr = mip == 0 ? psnr : mip0_psnr;
			if ( psnr < worst_psnr )
			{
				worst_psnr = psnr;
				worst_mip = mip;
				worst_offset = offset;
			}
			chain_error += error;
			chain_samples += samples;
			offset += size;
		}

		std::print( "{:<28} {:<9} {:>4} {:>9.2f} {:>9.2f} {:>9.2f} {:>5} {:>9.2f}\n",
					image_name,
					mode.name,
					refinements,
					mip0_psnr,
					to_psnr( chain_error, chain_samples ),
					worst_psnr,
					worst_mip,
					elapsed.count() );

		if ( refinements == 2 || !mode.uses_refinements )
		{
			save_png( out_dir / std::format( "{}_{}.png", image_name, mode.name ), readback.get_mapped_address(), extent );
			// Both sides of the worst mip, to look at what the encoder couldn't represent
			const auto worst_extent = src.get_extent( worst_mip );
			const auto expected_bytes = static_cast<uint8_t*>( source.readback.get_mapped_address() );
			const auto actual_bytes = static_cast<uint8_t*>( readback.get_mapped_address() );
			save_png( out_dir / std::format( "{}_{}_worst_source.png", image_name, mode.name ),
					  expected_bytes + worst_offset,
					  worst_extent );
			save_png( out_dir / std::format( "{}_{}_worst_decoded.png", image_name, mode.name ),
					  actual_bytes + worst_offset,
					  worst_extent );
		}
	}

	// Every 4x4 block holds one gray level: BC4 stores a constant block's value exactly, so the blocks show whether the
	// shader's sRGB re-encoding gets the 8 bit values back
	int count_srgb_round_trip_errors( renderer::Device& device,
									  Bindless& bindless,
									  const renderer::PipelineManager& pipelines,
									  renderer::PipelineHandle bc5_srgb_pipeline )
	{
		constexpr int VALUES = 256;
		constexpr int BLOCK_SIDE = 4;
		const renderer::Extent2D extent { VALUES * BLOCK_SIDE, BLOCK_SIDE };
		std::vector<uint8_t> pixels( static_cast<std::size_t>( extent.width ) * extent.height * RGBA_SIZE );
		for ( std::size_t y = 0; y < extent.height; ++y )
		{
			for ( std::size_t x = 0; x < extent.width; ++x )
			{
				const auto pixel = pixels.data() + ( y * extent.width + x ) * RGBA_SIZE;
				pixel[ 0 ] = pixel[ 1 ] = pixel[ 2 ] = static_cast<uint8_t>( x / BLOCK_SIDE );
				pixel[ 3 ] = 255;
			}
		}

		const auto source = make_source( device, bindless, pixels.data(), extent, Format::R8G8B8A8_SRGB, 1 );
		constexpr std::size_t BC5_BLOCK_SIZE = 16;
		const std::size_t size = VALUES * BC5_BLOCK_SIZE;
		auto blocks = make_block_buffer( device, size );
		auto readback = make_readback( device, size );
		const auto& pipeline = pipelines.get( bc5_srgb_pipeline );
		record_and_wait( device,
						 [ & ]( renderer::CommandBuffer& cmd )
						 {
							 cmd.bind_pipeline( pipeline, bindless );
							 dispatch_mip( cmd, pipeline, source.texture, blocks, 0, 0, 0 );
							 cmd.barrier( renderer::Stage::COMPUTE, renderer::Stage::TRANSFER );
							 cmd.copy_buffer( blocks, 0, size, readback );
							 cmd.barrier( renderer::Stage::TRANSFER, renderer::Stage::HOST );
						 } );
		readback.flush();

		const auto bytes = static_cast<const uint8_t*>( readback.get_mapped_address() );
		int errors = 0;
		for ( std::size_t value = 0; value < VALUES; ++value )
		{
			// A BC5 block starts with the two endpoints of its red channel
			const auto block = bytes + value * BC5_BLOCK_SIZE;
			if ( block[ 0 ] != value || block[ 1 ] != value )
			{
				std::print( "sRGB round trip: {} became {} / {}\n", value, block[ 0 ], block[ 1 ] );
				++errors;
			}
		}
		return errors;
	}
}

int main( int argc, char** argv )
{
	if ( argc < 3 )
	{
		std::print( stderr, "Usage: bc-compress-test <output directory> <image>...\n" );
		return EINVAL;
	}
	const std::filesystem::path out_dir = argv[ 1 ];
	std::filesystem::create_directories( out_dir );

	renderer::Device device( "bc-compress-test", renderer::Device::WindowMode::HIDDEN );
	Bindless bindless( device, std::array<uint32_t, 1> { 4096 } );
	renderer::PipelineManager pipelines( device, "shaders", bindless );

	const auto add_pipeline = [ & ]( const char* bc_format, const char* srgb )
	{
		const std::vector<renderer::ShaderSource::Define> defines { { "BC_FORMAT", bc_format }, { "SRGB", srgb } };
		return pipelines.add( renderer::Pipeline::Desc { .push_constants_size = sizeof( CompressConstants ) },
							  { { .path = "bc_compress.comp", .stage = renderer::ShaderStage::COMPUTE, .defines = defines } } );
	};

	std::array modes { Mode { "BC1_SRGB", "1", "1", Format::BC1_RGB_SRGB, Format::R8G8B8A8_SRGB, 3, true },
					   Mode { "BC1_UNORM", "1", "0", Format::BC1_RGB_UNORM, Format::R8G8B8A8_UNORM, 3, true },
					   Mode { "BC3_SRGB", "3", "1", Format::BC3_SRGB, Format::R8G8B8A8_SRGB, 4, true },
					   Mode { "BC3_UNORM", "3", "0", Format::BC3_UNORM, Format::R8G8B8A8_UNORM, 4, true },
					   Mode { "BC5_UNORM", "5", "0", Format::BC5_UNORM, Format::R8G8B8A8_UNORM, 2, false } };
	for ( auto& mode : modes )
	{
		mode.pipeline = add_pipeline( mode.bc_format, mode.srgb );
	}
	const auto bc5_srgb_pipeline = add_pipeline( "5", "1" );
	pipelines.wait_ready();

	const int round_trip_errors = count_srgb_round_trip_errors( device, bindless, pipelines, bc5_srgb_pipeline );
	std::print( "sRGB round trip errors: {} / 256\n\n", round_trip_errors );

	std::print( "{:<28} {:<9} {:>4} {:>9} {:>9} {:>9} {:>5} {:>9}\n",
				"image",
				"mode",
				"ref",
				"mip0 dB",
				"chain dB",
				"worst dB",
				"mip",
				"ms" );

	for ( int i = 2; i < argc; ++i )
	{
		const std::filesystem::path path = argv[ i ];
		const sdl::raii::Surface loaded( IMG_Load( path.string().c_str() ) );
		if ( !loaded )
		{
			throw renderer::Error( SDL_GetError() );
		}
		const sdl::raii::Surface surface( SDL_ConvertSurface( loaded.get(), SDL_PIXELFORMAT_RGBA32 ) );
		if ( !surface )
		{
			throw renderer::Error( SDL_GetError() );
		}
		assert( surface->pitch == surface->w * RGBA_SIZE );
		const renderer::Extent2D extent { static_cast<uint32_t>( surface->w ), static_cast<uint32_t>( surface->h ) };
		if ( !std::has_single_bit( extent.width ) || !std::has_single_bit( extent.height ) )
		{
			std::print( "{}: skipped, not a power of two ({}x{})\n", path.string(), extent.width, extent.height );
			continue;
		}

		const int mips = std::bit_width( std::max( extent.width, extent.height ) );
		const auto srgb = make_source( device, bindless, surface->pixels, extent, Format::R8G8B8A8_SRGB, mips );
		const auto unorm = make_source( device, bindless, surface->pixels, extent, Format::R8G8B8A8_UNORM, mips );
		for ( const auto& mode : modes )
		{
			const auto& source = mode.source == Format::R8G8B8A8_SRGB ? srgb : unorm;
			compress_and_measure( device, bindless, pipelines, mode, source, 2, path.stem().string(), out_dir );
			if ( mode.uses_refinements )
			{
				compress_and_measure( device, bindless, pipelines, mode, source, 0, path.stem().string(), out_dir );
			}
		}
		save_png( out_dir / std::format( "{}_source.png", path.stem().string() ), surface->pixels, extent );
	}

	device.wait_idle();
	PROFILER_SHUTDOWN();
	return round_trip_errors == 0 ? 0 : 1;
}
