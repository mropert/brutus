#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3_image/SDL_image.h>
#include <array>
#include <chrono>
#include <expected>
#include <fastgltf/core.hpp>
#include <filesystem>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_aligned.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <memory>
#include <meshoptimizer.h>
#include <numbers>
#include <optick.h>
#include <print>
#include <span>
#include <stdexcept>
#include <tbb/tbb.h>

//

#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/command_buffer.h>
#include <renderer/device.h>
#include <renderer/pipeline_manager.h>
#include <renderer/sampler.h>
#include <renderer/shader.h>
#include <renderer/swapchain.h>
#include <renderer/texture.h>

struct pbr_data
{
	glm::vec4 base_color;
	float metalness;
	float roughness;
};

struct gpu_material
{
	pbr_data pbr_data;
	renderer::TextureHandle albedo_texture_index;
	renderer::TextureHandle normal_texture_index;
};

struct push_constants
{
	gpu_material material;
	renderer::BufferHandle scene_constants;
	renderer::BufferHandle vertex_buffer;
	renderer::BufferHandle instance_buffer;
	renderer::BufferHandle meshlets_buffer;
	renderer::BufferHandle meshlets_vertices_buffer;
	renderer::BufferHandle meshlets_triangles_buffer;
};

SDL_PixelFormat sdl_gpu_format( SDL_PixelFormat format )
{
	switch ( format )
	{
		case SDL_PIXELFORMAT_BGR565:
			return SDL_PIXELFORMAT_BGR565;
		case SDL_PIXELFORMAT_BGRA5551:
			return SDL_PIXELFORMAT_BGRA5551;
		case SDL_PIXELFORMAT_BGR24:
		case SDL_PIXELFORMAT_BGRX32:
			return SDL_PIXELFORMAT_BGRX32;
		case SDL_PIXELFORMAT_BGRA32:
			return SDL_PIXELFORMAT_BGRA32;
		case SDL_PIXELFORMAT_RGB24:
		case SDL_PIXELFORMAT_RGBX32:
			return SDL_PIXELFORMAT_RGBX32;
		case SDL_PIXELFORMAT_RGBA32:
			return SDL_PIXELFORMAT_RGBA32;
		default:
			return SDL_PIXELFORMAT_UNKNOWN;
	}
}

VkFormat sdl_to_vulkan_format( SDL_PixelFormat format, bool sRGB = false )
{
	switch ( format )
	{
		case SDL_PIXELFORMAT_BGR565:
			return VK_FORMAT_B5G6R5_UNORM_PACK16;
		case SDL_PIXELFORMAT_BGRA5551:
			return VK_FORMAT_B5G5R5A1_UNORM_PACK16;
		case SDL_PIXELFORMAT_BGRA32:
		case SDL_PIXELFORMAT_BGRX32:
			return sRGB ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
		case SDL_PIXELFORMAT_RGBA32:
		case SDL_PIXELFORMAT_RGBX32:
			return sRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
		default:
			return VK_FORMAT_UNDEFINED;
	}
}

std::size_t get_format_size( VkFormat format )
{
	switch ( format )
	{
		case VK_FORMAT_B5G6R5_UNORM_PACK16:
		case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
			return 2;
		case VK_FORMAT_B8G8R8A8_UNORM:
		case VK_FORMAT_R8G8B8A8_UNORM:
			return 4;
		default:
			return 0;
	}
}

struct mesh_attribute
{
	std::size_t offset = 0;
	std::size_t stride = 0;
	std::size_t length = 0;
	VkFormat format = VK_FORMAT_UNDEFINED;
};

struct mesh_texture
{
	fastgltf::MimeType type = fastgltf::MimeType::None;
	std::size_t offset = 0;
	std::size_t size = 0;
	sdl::raii::Surface image;
};

struct mesh
{
	mesh_attribute positions;
	mesh_attribute normals;
	mesh_attribute tangents;
	mesh_attribute uvs;
	mesh_attribute indices;
	int albedo_idx = -1;
	int normal_idx = -1;
	pbr_data pbr_data;
};

struct mesh_vertex
{
	glm::vec3 position;
	float u;
	glm::vec3 normal;
	float v;
	glm::vec4 tangent;
};

struct gpu_mesh_data
{
	renderer::raii::Buffer vertices;
	renderer::raii::Buffer indices;
	renderer::raii::Buffer meshlets;
	renderer::raii::Buffer meshlets_vertices;
	renderer::raii::Buffer meshlets_triangles;
	std::array<uint32_t, 4> indices_count_for_lod;
	std::array<uint32_t, 4> meshlet_count_for_lod;
	std::array<uint32_t, 4> meshlet_vertices_count_for_lod;
	std::array<uint32_t, 4> meshlet_triangles_count_for_lod;
	gpu_material material;
};

renderer::raii::Texture make_image_from_surface( renderer::Device& device, const SDL_Surface& surface, bool sRGB )
{
	OPTICK_EVENT();
	const auto format = sdl_to_vulkan_format( sdl_gpu_format( surface.format ), sRGB );
	if ( format == VK_FORMAT_UNDEFINED )
	{
		throw renderer::Error( "Unsupported format" );
	}
	using TexUsage = renderer::Texture::Usage;
	return device.create_texture( static_cast<renderer::Texture::Format>( format ),
								  TexUsage::TRANSFER_DST | TexUsage::SAMPLED,
								  renderer::Extent2D( surface.w, surface.h ) );
}

std::size_t upload_image_from_surface( renderer::CommandBuffer& cmd,
									   const renderer::Buffer& transfer_buffer,
									   std::byte* buffer,
									   std::size_t offset,
									   renderer::Texture& image,
									   const SDL_Surface& surface )
{
	OPTICK_EVENT();
	const auto gpu_format = sdl_gpu_format( surface.format );
	const auto pitch = surface.w * get_format_size( sdl_to_vulkan_format( gpu_format ) );
	SDL_ConvertPixels( surface.w, surface.h, surface.format, surface.pixels, surface.pitch, gpu_format, buffer + offset, pitch );
	cmd.transition_texture( image, renderer::Texture::Layout::TRANSFER_DST_OPTIMAL );
	cmd.copy_buffer_to_texture( transfer_buffer, offset, image );
	cmd.transition_texture( image, renderer::Texture::Layout::SHADER_READ_ONLY_OPTIMAL );
	return pitch * surface.h;
}

class mesh_load_job
{
public:
	mesh_load_job( renderer::Device& device, renderer::BindlessManager& tex_manager, std::filesystem::path path )
		: _device( device )
		, _tex_manager( tex_manager )
		, _path( std::move( path ) )
		, _vertex_cmd_buffer( device.grab_command_buffer() )
		, _texture_cmd_buffer( device.grab_command_buffer() )
		, _vertex_upload_fence( device.create_fence() )
		, _tex_upload_fence( device.create_fence() )
	{
	}

	void parse()
	{
		OPTICK_EVENT();

		fastgltf::Parser parser;
		fastgltf::GltfDataBuffer data;
		{
			OPTICK_EVENT( "gltf_read_io" );
			auto res = fastgltf::GltfDataBuffer::FromPath( _path );
			if ( res.error() != fastgltf::Error::None )
			{
				return;
			}
			data = std::move( res.get() );
		}

		{
			OPTICK_EVENT( "gltf_parse" );
			auto res = parser.loadGltf( data, _path.parent_path(), fastgltf::Options::None );
			if ( auto error = res.error(); error != fastgltf::Error::None )
			{
				return;
			}
			_asset = std::move( res.get() );
		}

		auto make_texture = []( const fastgltf::Asset& asset, std::size_t texture_index )
		{
			const auto& texture = asset.textures[ texture_index ];
			const auto& image = asset.images[ *texture.imageIndex ];
			const auto& data = std::get<fastgltf::sources::BufferView>( image.data );
			const auto& view = asset.bufferViews[ data.bufferViewIndex ];
			return mesh_texture { data.mimeType, view.byteOffset, view.byteLength };
		};

		auto make_attribute = []( const fastgltf::Asset& asset, std::size_t accessor_index )
		{
			const auto& accessor = asset.accessors[ accessor_index ];
			const auto& view = asset.bufferViews[ *accessor.bufferViewIndex ];
			VkFormat format = VK_FORMAT_UNDEFINED;
			switch ( accessor.componentType )
			{
				case fastgltf::ComponentType::Float:
				{
					switch ( accessor.type )
					{
						case fastgltf::AccessorType::Scalar:
							format = VK_FORMAT_R32_SFLOAT;
							break;
						case fastgltf::AccessorType::Vec2:
							format = VK_FORMAT_R32G32_SFLOAT;
							break;
						case fastgltf::AccessorType::Vec3:
							format = VK_FORMAT_R32G32B32_SFLOAT;
							break;
						case fastgltf::AccessorType::Vec4:
							format = VK_FORMAT_R32G32B32A32_SFLOAT;
							break;
					}
					break;
				}
				case fastgltf::ComponentType::UnsignedInt:
				{
					switch ( accessor.type )
					{
						case fastgltf::AccessorType::Scalar:
							format = VK_FORMAT_R32_UINT;
							break;
						case fastgltf::AccessorType::Vec2:
							format = VK_FORMAT_R32G32_UINT;
							break;
						case fastgltf::AccessorType::Vec3:
							format = VK_FORMAT_R32G32B32_UINT;
							break;
						case fastgltf::AccessorType::Vec4:
							format = VK_FORMAT_R32G32B32A32_UINT;
							break;
					}
					break;
				}
			}
			return mesh_attribute { .offset = accessor.byteOffset + view.byteOffset,
									.stride = view.byteStride.value_or( 0 ),
									.length = accessor.count,
									.format = format };
		};

		for ( const auto& mesh : _asset.meshes )
		{
			auto& m = _meshes.emplace_back();
			if ( mesh.primitives[ 0 ].materialIndex )
			{
				const auto& material = _asset.materials[ *mesh.primitives[ 0 ].materialIndex ];
				if ( material.pbrData.baseColorTexture )
				{
					_textures.push_back( make_texture( _asset, material.pbrData.baseColorTexture->textureIndex ) );
					m.albedo_idx = _textures.size() - 1;
				}
				if ( material.normalTexture )
				{
					_textures.push_back( make_texture( _asset, material.normalTexture->textureIndex ) );
					m.normal_idx = _textures.size() - 1;
				}
				m.pbr_data.base_color.x = material.pbrData.baseColorFactor.x();
				m.pbr_data.base_color.y = material.pbrData.baseColorFactor.y();
				m.pbr_data.base_color.z = material.pbrData.baseColorFactor.z();
				m.pbr_data.base_color.w = material.pbrData.baseColorFactor.w();
				m.pbr_data.metalness = material.pbrData.metallicFactor;
				m.pbr_data.roughness = material.pbrData.roughnessFactor;
			}

			for ( const auto& attribute : mesh.primitives[ 0 ].attributes )
			{
				if ( attribute.name == "POSITION" )
				{
					m.positions = make_attribute( _asset, attribute.accessorIndex );
				}
				else if ( attribute.name == "NORMAL" )
				{
					m.normals = make_attribute( _asset, attribute.accessorIndex );
				}
				else if ( attribute.name == "TANGENT" )
				{
					m.tangents = make_attribute( _asset, attribute.accessorIndex );
				}
				else if ( attribute.name == "TEXCOORD_0" )
				{
					m.uvs = make_attribute( _asset, attribute.accessorIndex );
				}
			}
			if ( mesh.primitives[ 0 ].indicesAccessor )
			{
				m.indices = make_attribute( _asset, *mesh.primitives[ 0 ].indicesAccessor );
			}
		}

		_gpu_meshes.resize( _meshes.size() );
	}

	struct lod_indices
	{
		std::vector<uint32_t> indices;
		std::array<uint32_t, 3> indices_per_lod;
	};

	void upload_vertices()
	{
		OPTICK_EVENT();
		const auto& gltf_buffer = std::get<fastgltf::sources::Array>( _asset.buffers[ 0 ].data ).bytes;

		std::vector<std::vector<mesh_vertex>> vertex_data( _meshes.size() );
		for ( int i = 0; i < _meshes.size(); ++i )
		{
			OPTICK_EVENT( "repack_vertices" );
			vertex_data[ i ].reserve( _meshes[ i ].positions.length );

			const auto positions = reinterpret_cast<const float*>( gltf_buffer.data() + _meshes[ i ].positions.offset );
			const auto normals = reinterpret_cast<const float*>( gltf_buffer.data() + _meshes[ i ].normals.offset );
			const auto tangents = reinterpret_cast<const float*>( gltf_buffer.data() + _meshes[ i ].tangents.offset );
			const auto uvs = reinterpret_cast<const float*>( gltf_buffer.data() + _meshes[ i ].uvs.offset );
			for ( int v = 0; v < _meshes[ i ].positions.length; ++v )
			{
				vertex_data[ i ].emplace_back(
					glm::vec3( positions[ v * 3 ], positions[ v * 3 + 1 ], positions[ v * 3 + 2 ] ),
					uvs[ v * 2 ],
					glm::vec3( normals[ v * 3 ], normals[ v * 3 + 1 ], normals[ v * 3 + 2 ] ),
					uvs[ v * 2 + 1 ],
					glm::vec4( tangents[ v * 4 ], tangents[ v * 4 + 1 ], tangents[ v * 4 + 2 ], tangents[ v * 4 + 3 ] ) );
			}
		}

		std::vector<lod_indices> simplified_indices( _meshes.size() );

		{
			OPTICK_EVENT( "lod_generation" );
			constexpr std::array<float, 3> lod_indices_factor { { 0.2, 0.05, 0.01 } };
			constexpr std::array<float, 3> lod_error_factor { { 0.01, 0.04, 0.1 } };
			std::vector<uint32_t> buffer;
			for ( int i = 0; i < _meshes.size(); ++i )
			{
				buffer.resize( _meshes[ i ].indices.length );
				const auto indices = reinterpret_cast<const uint32_t*>( gltf_buffer.data() + _meshes[ i ].indices.offset );
				for ( int lod = 0; lod < simplified_indices[ i ].indices_per_lod.size(); ++lod )
				{
					const float attr_weights[ 3 ] = { 0.5f, 0.5f, 0.5f };
					const uint32_t size = meshopt_simplifyWithAttributes( buffer.data(),
																		  indices,
																		  _meshes[ i ].indices.length,
																		  &vertex_data[ i ].front().position.x,
																		  vertex_data[ i ].size(),
																		  sizeof( mesh_vertex ),
																		  &vertex_data[ i ].front().normal.x,
																		  sizeof( mesh_vertex ),
																		  attr_weights,
																		  3,
																		  nullptr,
																		  _meshes[ i ].indices.length * lod_indices_factor[ lod ],
																		  lod_error_factor[ lod ],
																		  0,
																		  nullptr );
					simplified_indices[ i ].indices_per_lod[ lod ] = size;
					simplified_indices[ i ].indices.insert( end( simplified_indices[ i ].indices ), buffer.data(), buffer.data() + size );
				}
			}
		}

		std::vector<std::vector<meshopt_Meshlet>> all_meshlets;
		std::vector<std::vector<uint32_t>> all_meshlet_vertices;
		std::vector<std::vector<uint8_t>> all_meshlet_triangles;
		std::vector<std::size_t> meshlets_counts( _meshes.size() );
		std::vector<std::size_t> meshlets_vertices_counts( _meshes.size() );
		std::vector<std::size_t> meshlets_triangles_counts( _meshes.size() );
		{
			OPTICK_EVENT( "meshlet_generation" );
			for ( int i = 0; i < _meshes.size(); ++i )
			{
				uint32_t index_offset = 0;
				for ( int lod = 0; lod < 4; ++lod )
				{
					const auto indices = lod == 0 ? reinterpret_cast<const uint32_t*>( gltf_buffer.data() + _meshes[ i ].indices.offset )
												  : simplified_indices[ i ].indices.data() + index_offset;
					const auto indices_count = lod == 0 ? _meshes[ i ].indices.length : simplified_indices[ i ].indices_per_lod[ lod - 1 ];

					auto max_meshlets = meshopt_buildMeshletsBound( indices_count, 64, 96 );
					auto& meshlets = all_meshlets.emplace_back( max_meshlets );
					auto& meshlet_vertices = all_meshlet_vertices.emplace_back( indices_count );
					auto& meshlet_triangles = all_meshlet_triangles.emplace_back( indices_count + max_meshlets * 3 );

					const auto meshlet_count = meshopt_buildMeshlets( meshlets.data(),
																	  meshlet_vertices.data(),
																	  meshlet_triangles.data(),
																	  indices,
																	  indices_count,
																	  &vertex_data[ i ].front().position.x,
																	  vertex_data[ i ].size(),
																	  sizeof( mesh_vertex ),
																	  64,
																	  96,
																	  0.f );

					const auto& last = meshlets[ meshlet_count - 1 ];

					meshlet_vertices.resize( last.vertex_offset + last.vertex_count );
					meshlet_triangles.resize( last.triangle_offset + last.triangle_count * 3 );
					meshlets.resize( meshlet_count );
					meshlets_counts[ i ] += meshlets.size();
					meshlets_vertices_counts[ i ] += meshlet_vertices.size();
					meshlets_triangles_counts[ i ] += meshlet_triangles.size();
				}
			}
		}

		std::size_t alloc_size = 0;
		for ( int i = 0; i < _meshes.size(); ++i )
		{
			alloc_size += sizeof( mesh_vertex ) * vertex_data[ i ].size();
			alloc_size += sizeof( uint32_t ) * ( _meshes[ i ].indices.length + simplified_indices[ i ].indices.size() );
			alloc_size += sizeof( meshopt_Meshlet ) * meshlets_counts[ i ];
			alloc_size += sizeof( uint32_t ) * meshlets_vertices_counts[ i ];
			alloc_size += sizeof( uint8_t ) * meshlets_triangles_counts[ i ];
		}

		_vertex_cmd_buffer->reset();
		_vertex_cmd_buffer->begin();

		using BufferUsage = renderer::Buffer::Usage;
		_vertex_upload_buffer = _device.create_buffer( BufferUsage::TRANSFER_SRC, alloc_size, true );
		const auto ptr = static_cast<std::byte*>( _vertex_upload_buffer.get_mapped_address() );

		std::size_t offset = 0;
		for ( int i = 0; i < _meshes.size(); ++i )
		{
			const auto& mesh = _meshes[ i ];
			auto& gpu_data = _gpu_meshes[ i ];
			gpu_data.vertices = _device.create_buffer( BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST
														   | BufferUsage::SHADER_DEVICE_ADDRESS,
													   sizeof( mesh_vertex ) * mesh.positions.length );

			gpu_data.indices = _device.create_buffer( BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
													  sizeof( uint32_t )
														  * ( _meshes[ i ].indices.length + simplified_indices[ i ].indices.size() ) );

			gpu_data.meshlets = _device.create_buffer( BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST
														   | BufferUsage::SHADER_DEVICE_ADDRESS,
													   meshlets_counts[ i ] * sizeof( meshopt_Meshlet ) );

			gpu_data.meshlets_vertices = _device.create_buffer( BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST
																	| BufferUsage::SHADER_DEVICE_ADDRESS,
																meshlets_vertices_counts[ i ] * sizeof( uint32_t ) );

			gpu_data.meshlets_triangles = _device.create_buffer( BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST
																	 | BufferUsage::SHADER_DEVICE_ADDRESS,
																 meshlets_triangles_counts[ i ] * sizeof( uint8_t ) );

			memcpy( ptr + offset, vertex_data[ i ].data(), sizeof( mesh_vertex ) * vertex_data[ i ].size() );
			_vertex_cmd_buffer->copy_buffer( _vertex_upload_buffer,
											 offset,
											 sizeof( mesh_vertex ) * vertex_data[ i ].size(),
											 gpu_data.vertices );
			offset += sizeof( mesh_vertex ) * vertex_data[ i ].size();

			memcpy( ptr + offset, gltf_buffer.data() + mesh.indices.offset, sizeof( uint32_t ) * mesh.indices.length );
			memcpy( ptr + offset + sizeof( uint32_t ) * mesh.indices.length,
					simplified_indices[ i ].indices.data(),
					sizeof( uint32_t ) * simplified_indices[ i ].indices.size() );
			const auto total_index_buffer_size = sizeof( uint32_t ) * ( mesh.indices.length + simplified_indices[ i ].indices.size() );
			_vertex_cmd_buffer->copy_buffer( _vertex_upload_buffer, offset, total_index_buffer_size, gpu_data.indices );
			offset += total_index_buffer_size;
			gpu_data.indices_count_for_lod[ 0 ] = mesh.indices.length;
			gpu_data.indices_count_for_lod[ 1 ] = simplified_indices[ i ].indices_per_lod[ 0 ];
			gpu_data.indices_count_for_lod[ 2 ] = simplified_indices[ i ].indices_per_lod[ 1 ];
			gpu_data.indices_count_for_lod[ 3 ] = simplified_indices[ i ].indices_per_lod[ 2 ];

			size_t lod_offset = 0zu;
			for ( int lod = 0; lod < 4; ++lod )
			{
				const auto count = all_meshlets[ i * 4 + lod ].size();
				const auto size = count * sizeof( meshopt_Meshlet );
				memcpy( ptr + offset + lod_offset, all_meshlets[ i * 4 + lod ].data(), size );
				lod_offset += size;
				gpu_data.meshlet_count_for_lod[ lod ] = count;
			}
			_vertex_cmd_buffer->copy_buffer( _vertex_upload_buffer, offset, lod_offset, gpu_data.meshlets );
			offset += lod_offset;

			lod_offset = 0zu;
			for ( int lod = 0; lod < 4; ++lod )
			{
				const auto count = all_meshlet_vertices[ i * 4 + lod ].size();
				const auto size = count * sizeof( uint32_t );
				memcpy( ptr + offset + lod_offset, all_meshlet_vertices[ i * 4 + lod ].data(), size );
				lod_offset += size;
				gpu_data.meshlet_vertices_count_for_lod[ lod ] = count;
			}
			_vertex_cmd_buffer->copy_buffer( _vertex_upload_buffer, offset, lod_offset, gpu_data.meshlets_vertices );
			offset += lod_offset;

			lod_offset = 0zu;
			for ( int lod = 0; lod < 4; ++lod )
			{
				const auto count = all_meshlet_triangles[ i * 4 + lod ].size();
				const auto size = count * sizeof( uint8_t );
				memcpy( ptr + offset + lod_offset, all_meshlet_triangles[ i * 4 + lod ].data(), size );
				lod_offset += size;
				gpu_data.meshlet_triangles_count_for_lod[ lod ] = count;
			}
			_vertex_cmd_buffer->copy_buffer( _vertex_upload_buffer, offset, lod_offset, gpu_data.meshlets_triangles );
			offset += lod_offset;
		}

		_vertex_cmd_buffer->end();
		_device.submit( *_vertex_cmd_buffer, _vertex_upload_fence );
	}

	void decompress_textures()
	{
		OPTICK_EVENT();

		0uz;

		const auto& gltf_buffer = std::get<fastgltf::sources::Array>( _asset.buffers[ 0 ].data ).bytes;
		tbb::parallel_for(
			0uz,
			_textures.size(),
			[ & ]( size_t index )
			{
				OPTICK_EVENT( "decompress_tex" );
				_textures[ index ].image.reset(
					IMG_Load_IO( SDL_IOFromConstMem( gltf_buffer.data() + _textures[ index ].offset, _textures[ index ].size ), true ) );
			} );
	}

	void upload_textures()
	{
		OPTICK_EVENT();

		std::size_t alloc_size = 0;
		for ( auto& tex : _textures )
		{
			alloc_size += tex.image->w * tex.image->h * get_format_size( sdl_to_vulkan_format( sdl_gpu_format( tex.image->format ) ) );
		}

		_texture_cmd_buffer->reset();
		_texture_cmd_buffer->begin();

		using BufferUsage = renderer::Buffer::Usage;
		_texture_upload_buffer = _device.create_buffer( BufferUsage::TRANSFER_SRC, alloc_size, true );
		const auto ptr = static_cast<std::byte*>( _texture_upload_buffer.get_mapped_address() );

		std::size_t offset = 0;
		for ( int i = 0; i < _meshes.size(); ++i )
		{
			const auto& mesh = _meshes[ i ];
			auto& gpu_data = _gpu_meshes[ i ];

			auto tex = make_image_from_surface( _device, *_textures[ mesh.albedo_idx ].image, true );
			offset += upload_image_from_surface( *_texture_cmd_buffer,
												 _texture_upload_buffer,
												 ptr,
												 offset,
												 tex,
												 *_textures[ mesh.albedo_idx ].image );
			gpu_data.material.albedo_texture_index = _tex_manager.add_texture( std::move( tex ) );
			tex = make_image_from_surface( _device, *_textures[ mesh.normal_idx ].image, false );
			offset += upload_image_from_surface( *_texture_cmd_buffer,
												 _texture_upload_buffer,
												 ptr,
												 offset,
												 tex,
												 *_textures[ mesh.normal_idx ].image );
			gpu_data.material.normal_texture_index = _tex_manager.add_texture( std::move( tex ) );
			gpu_data.material.pbr_data = mesh.pbr_data;
		}

		_texture_cmd_buffer->end();
		_device.submit( *_texture_cmd_buffer, _tex_upload_fence );
	}

	std::vector<gpu_mesh_data> finalize()
	{
		OPTICK_EVENT();
		_device.wait_for_fences( { _vertex_upload_fence, _tex_upload_fence }, 9999999999 );
		return std::move( _gpu_meshes );
	}

	std::size_t get_texture_count() const { return _textures.size(); }

private:
	renderer::Device& _device;
	renderer::BindlessManager& _tex_manager;
	std::filesystem::path _path;
	renderer::raii::CommandBuffer _vertex_cmd_buffer;
	renderer::raii::CommandBuffer _texture_cmd_buffer;
	renderer::raii::Buffer _vertex_upload_buffer;
	renderer::raii::Buffer _texture_upload_buffer;
	renderer::raii::Fence _vertex_upload_fence;
	renderer::raii::Fence _tex_upload_fence;
	fastgltf::Asset _asset;
	std::vector<mesh_texture> _textures;
	std::vector<mesh> _meshes;
	std::vector<gpu_mesh_data> _gpu_meshes;
};

std::vector<gpu_mesh_data> load_model( renderer::Device& device, renderer::BindlessManager& tex_manager )
{
	OPTICK_EVENT();
	mesh_load_job job( device, tex_manager, "../assets/player.glb" );
	job.parse();
	tbb::parallel_invoke( [ & ]() { job.upload_vertices(); },
						  [ & ]
						  {
							  job.decompress_textures();
							  job.upload_textures();
						  } );
	return job.finalize();
}

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
	glm::mat4 view_matrix;
	glm::mat4 inverse_view_matrix;
	glm::aligned_vec3 camera_position;
	glm::vec4 main_light;
};

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
	return scene_constants { .view_projection_matrix = projection * view,
							 .view_matrix = view,
							 .inverse_view_matrix = glm::inverse( view ),
							 .camera_position = camera.position,
							 .main_light = main_light };
}

struct ui_data
{
	int x_miku = 1;
	int z_miku = 1;
	float lod_cutoff = 10.f;
	bool paused = false;
	bool vsync;
	bool mesh_shading;
};

void draw_ui( ui_data& data,
			  const renderer::Device::Properties& props,
			  std::span<const gpu_mesh_data> meshes,
			  float gpu_time,
			  std::span<float> lod_times )
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
		ImGui::SeparatorText( "Stats" );
		ImGui::Text( "GPU time(ms): %.2f", gpu_time );
		ImGui::Text( "Per LOD GPU time(ms): %.2f / %.2f / %.2f / %.2f", lod_times[ 0 ], lod_times[ 1 ], lod_times[ 2 ], lod_times[ 3 ] );
		for ( int i = 0; i < meshes.size(); ++i )
		{
			ImGui::Text( "Mesh #%d: %d / %d / %d / %d triangles",
						 i,
						 meshes[ i ].indices_count_for_lod[ 0 ],
						 meshes[ i ].indices_count_for_lod[ 1 ],
						 meshes[ i ].indices_count_for_lod[ 2 ],
						 meshes[ i ].indices_count_for_lod[ 3 ] );
		}

		ImGui::SeparatorText( "Settings" );
		if ( ImGui::Button( data.paused ? "Unpause" : "Pause" ) )
		{
			data.paused = !data.paused;
		}
		ImGui::Checkbox( "V-Sync", &data.vsync );
		ImGui::Checkbox( "Mesh shading", &data.mesh_shading );
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

namespace Timestamps
{
	enum : uint32_t
	{
		FRAME_START = 0,
		DRAW_MESHES,
		DRAW_GRID = DRAW_MESHES + 64,
		DRAW_UI,
		BLIT,
		FRAME_END,
		COUNT
	};
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
										 .MSAASamples = VK_SAMPLE_COUNT_4_BIT,
										 .DescriptorPoolSize = renderer::MAX_FRAMES_IN_FLIGHT,
										 .UseDynamicRendering = true,
										 .PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
																		  .colorAttachmentCount = 1,
																		  .pColorAttachmentFormats = &color_format,
																		  .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT } };
		ImGui_ImplVulkan_Init( &info );
	}

	const auto draw_image_extent = device.get_extent();
	using usage = renderer::Texture::Usage;
	auto draw_image = device.create_texture( draw_image_format, usage::COLOR_ATTACHMENT, draw_image_extent, 4 );
	auto draw_image_view = device.create_texture_view( draw_image, renderer::TextureView::Aspect::COLOR );

	auto depth_image = device.create_texture( renderer::Texture::Format::D32_SFLOAT,
											  usage::DEPTH_STENCIL_ATTACHMENT,
											  draw_image_extent,
											  4 );
	auto depth_image_view = device.create_texture_view( depth_image, renderer::TextureView::Aspect::DEPTH );

	auto draw_resolved_image = device.create_texture( draw_image_format, usage::TRANSFER_SRC | usage::COLOR_ATTACHMENT, draw_image_extent );
	auto draw_resolved_image_view = device.create_texture_view( draw_resolved_image, renderer::TextureView::Aspect::COLOR );

	renderer::BindlessManager tex_manager( device );
	renderer::PipelineManager pipelines( device, "../shaders/vulkan", tex_manager );

	renderer::PipelineHandle mesh_pipeline;
	renderer::PipelineHandle mesh_pipeline_meshlet;
	renderer::PipelineHandle grid_pipeline;
	std::vector<gpu_mesh_data> meshes;
	{
		renderer::ShaderCompiler compiler( "../shaders/vulkan" );
		tbb::parallel_invoke(
			[ & ]
			{
				mesh_pipeline = pipelines.add(
					renderer::Pipeline::Desc { .color_format = draw_image_format,
											   .depth_format = depth_image.get_format(),
											   .push_constants_size = sizeof( push_constants ) },
					{ { "mesh.vert", renderer::ShaderStage::VERTEX }, { "mesh.frag", renderer::ShaderStage::FRAGMENT } } );
			},
			[ & ]
			{
				mesh_pipeline_meshlet = pipelines.add(
					renderer::Pipeline::Desc { .color_format = draw_image_format,
											   .depth_format = depth_image.get_format(),
											   .push_constants_size = sizeof( push_constants ) },
					{ { "mesh.mesh", renderer::ShaderStage::MESH }, { "mesh.frag", renderer::ShaderStage::FRAGMENT } } );
			},
			[ & ]
			{
				grid_pipeline = pipelines.add(
					renderer::Pipeline::Desc { .color_format = draw_image_format,
											   .depth_format = depth_image.get_format(),
											   .topology = renderer::Pipeline::PrimitiveTopology::TRIANGLE_STRIP,
											   .push_constants_size = sizeof( grid_constants ) },
					{ { "grid.vert", renderer::ShaderStage::VERTEX }, { "grid.frag", renderer::ShaderStage::FRAGMENT } } );
			},
			[ & ] { meshes = load_model( device, tex_manager ); } );
	}

	auto constant_buffers = device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
												  sizeof( std::array<scene_constants, renderer::MAX_FRAMES_IN_FLIGHT> ),
												  true );
	const auto constant_buffers_ptr = static_cast<scene_constants*>( constant_buffers.get_mapped_address() );

	static const uint32_t max_instances = 128 * 128;
	using Instances = std::array<glm::mat4, max_instances>;
	auto instance_buffers = device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
												  sizeof( Instances ) * renderer::MAX_FRAMES_IN_FLIGHT,
												  true );
	const auto instance_buffers_ptr = static_cast<Instances*>( instance_buffers.get_mapped_address() );

	const auto timestamp_period = device.get_timestamp_period();
	std::array<renderer::raii::QueryPool, renderer::MAX_FRAMES_IN_FLIGHT> query_pools { device.create_query_pool( Timestamps::COUNT ),
																						device.create_query_pool( Timestamps::COUNT ) };

	OPTICK_POP();

	Camera camera { .position = glm::vec3( 0, 1.f, 2.f ) };

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
	ui_data ui_state { .vsync = vsync, .mesh_shading = props.mesh_shader_support };

	std::array<std::vector<glm::mat4>, 4> instances_per_lod;

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

		const auto constants = build_scene_constants( camera, draw_image_extent, time_of_day / day_period );

		for ( auto& instances : instances_per_lod )
		{
			instances.clear();
		}

		for ( int z = 0; z < ui_state.z_miku; ++z )
		{
			for ( int x = 0; x < ui_state.x_miku; ++x )
			{
				const auto pos = glm::vec3( x - ( ui_state.x_miku / 2 ), 0, z - ( ui_state.z_miku / 2 ) );
				const auto distance = glm::distance( camera.position, pos );
				int lod = instances_per_lod.size() - 1;
				while ( distance < ui_state.lod_cutoff * lod )
				{
					--lod;
				}
				instances_per_lod[ lod ].push_back( glm::translate( glm::identity<glm::mat4>(), pos ) );
			}
		}

		auto [ image_index, swapchain_image, swapchain_image_view ] = swapchain.acquire();

		const uint32_t query_pool_offset = image_index * Timestamps::COUNT;
		std::array<uint64_t, Timestamps::COUNT> timestamps;
		float gpu_time = 0.f;
		std::array<float, 4> per_lod_gpu_time {};
		if ( frame_count > renderer::MAX_FRAMES_IN_FLIGHT )
		{
			device.get_query_pool_results( query_pools[ image_index ], 0, timestamps );
			gpu_time = ( timestamps[ Timestamps::FRAME_END ] - timestamps[ Timestamps::FRAME_START ] ) * timestamp_period * 1e-6f;
			for ( int lod = 0; lod < 4; ++lod )
			{
				uint64_t time = 0;
				for ( int mesh = 0; mesh < meshes.size(); ++mesh )
				{
					time += timestamps[ Timestamps::DRAW_MESHES + ( mesh * 4 ) + lod + 1 ]
						- timestamps[ Timestamps::DRAW_MESHES + ( mesh * 4 ) + lod ];
				}
				per_lod_gpu_time[ lod ] = time * timestamp_period * 1e-6f;
			}
		}

		memcpy( constant_buffers_ptr + image_index, &constants, sizeof( constants ) );
		const auto constants_address = constant_buffers.get_device_address() + ( image_index * sizeof( constants ) );

		auto instance_buffer = ( instance_buffers_ptr + image_index )->data();
		for ( const auto& instances : instances_per_lod )
		{
			if ( !instances.empty() )
			{
				memcpy( instance_buffer, instances.data(), instances.size() * sizeof( glm::mat4 ) );
				instance_buffer += instances.size();
			}
		}
		const auto instances_address = instance_buffers.get_device_address() + ( image_index * sizeof( Instances ) );

		pipelines.update();

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();

		draw_ui( ui_state, props, meshes, gpu_time, per_lod_gpu_time );

		auto command_buffer = device.grab_command_buffer();
		command_buffer->reset();
		command_buffer->begin();

		command_buffer->reset_query_pool( query_pools[ image_index ], 0, Timestamps::COUNT );
		command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::FRAME_START );

		command_buffer->transition_texture( draw_image, renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( draw_resolved_image, renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( depth_image, renderer::Texture::Layout::DEPTH_ATTACHMENT_OPTIMAL );

		command_buffer->begin_rendering( draw_image_extent,
										 renderer::RenderAttachment { .target = draw_image_view,
																	  .resolve_target = draw_resolved_image_view,
																	  .clear_value = { { 0.1f, 0.1f, 0.1f, 1.f } } },
										 renderer::RenderAttachment { .target = depth_image_view } );

		command_buffer->set_viewport( draw_image_extent );
		command_buffer->set_scissor( draw_image_extent );

		int mesh_draws = 0;

		if ( ui_state.mesh_shading )
		{
			command_buffer->bind_pipeline( pipelines.get( mesh_pipeline_meshlet ), tex_manager );
			for ( const auto& mesh : meshes )
			{
				OPTICK_GPU_EVENT( "Draw Meshlet" );
				const push_constants constants { .material = mesh.material,
												 .scene_constants = constants_address,
												 .vertex_buffer = mesh.vertices.get_device_address(),
												 .instance_buffer = instances_address,
												 .meshlets_buffer = mesh.meshlets.get_device_address(),
												 .meshlets_vertices_buffer = mesh.meshlets_vertices.get_device_address(),
												 .meshlets_triangles_buffer = mesh.meshlets_triangles.get_device_address() };
				command_buffer->push_constants( pipelines.get( mesh_pipeline_meshlet ), constants );
				command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_MESHES + mesh_draws );
				const auto max_draws = static_cast<std::size_t>( props.max_mesh_shader_groups ) / mesh.meshlet_count_for_lod[ 0 ];
				const auto count = std::min( max_draws, instances_per_lod[ 0 ].size() );
				command_buffer->draw_mesh_task( mesh.meshlet_count_for_lod[ 0 ], count, 1 );
				++mesh_draws;
				for ( int i = 0; i < 3; ++i )
				{
					command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_MESHES + mesh_draws );
					++mesh_draws;
				}
			}
		}
		else
		{
			command_buffer->bind_pipeline( pipelines.get( mesh_pipeline ), tex_manager );
			for ( const auto& mesh : meshes )
			{
				OPTICK_GPU_EVENT( "Draw Mesh" );
				command_buffer->bind_index_buffer( mesh.indices );
				const push_constants constants { .material = mesh.material,
												 .scene_constants = constants_address,
												 .vertex_buffer = mesh.vertices.get_device_address(),
												 .instance_buffer = instances_address };

				command_buffer->push_constants( pipelines.get( mesh_pipeline ), constants );
				uint32_t base_index = 0;
				uint32_t base_instance = 0;
				for ( int lod = 0; lod < instances_per_lod.size(); ++lod )
				{
					command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_MESHES + mesh_draws );
					if ( !instances_per_lod[ lod ].empty() )
					{
						command_buffer->draw_indexed( mesh.indices_count_for_lod[ lod ],
													  instances_per_lod[ lod ].size(),
													  base_index,
													  base_instance );
						base_instance += instances_per_lod[ lod ].size();
					}
					base_index += mesh.indices_count_for_lod[ lod ];
					++mesh_draws;
				}
			}
		}
		command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_MESHES + mesh_draws );

		{
			OPTICK_GPU_EVENT( "Draw Grid" );
			command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_GRID );
			command_buffer->bind_pipeline( pipelines.get( grid_pipeline ), tex_manager );
			command_buffer->push_constants( pipelines.get( grid_pipeline ), grid_constants { .scene_constants = constants_address } );
			command_buffer->draw( 4 );
		}

		{
			OPTICK_GPU_EVENT( "Draw UI" );
			command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_UI );
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData( ImGui::GetDrawData(), command_buffer->get_impl() );
		}

		command_buffer->end_rendering();

		command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::BLIT );


		command_buffer->transition_texture( draw_resolved_image, renderer::Texture::Layout::TRANSFER_SRC_OPTIMAL );
		command_buffer->transition_texture( swapchain_image, renderer::Texture::Layout::TRANSFER_DST_OPTIMAL );

		command_buffer->blit_texture( draw_resolved_image, swapchain_image );

		command_buffer->transition_texture( swapchain_image, renderer::Texture::Layout::PRESENT_SRC );

		command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::FRAME_END );

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
	}

	// Wait for the GPU to finish before we roll all the destructors
	device.wait_idle();

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();

	OPTICK_SHUTDOWN();

	return 0;
}
