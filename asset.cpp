#include "asset.h"

#include <SDL3_image/SDL_image.h>
#include <fastgltf/core.hpp>
#include <meshoptimizer.h>
#include <mikktspace.h>
#include <numeric>
#include <optick.h>
#include <renderer/command_buffer.h>
#include <renderer/texture.h>
#include <renderer/third_party/tbb.h>

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

struct mesh_vertex
{
	glm::vec3 position;
	float u;
	glm::vec3 normal;
	float v;
	glm::vec4 tangent;
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
	return device.create_texture( renderer::Texture::Desc { static_cast<renderer::Texture::Format>( format ),
															TexUsage::TRANSFER_DST | TexUsage::SAMPLED,
															renderer::Extent2D( surface.w, surface.h ) } );
}

namespace MikkTSpace
{
	struct UserData
	{
		std::span<const glm::vec3> positions;
		std::span<const glm::vec3> normals;
		std::span<const glm::vec2> uvs;
		std::span<glm::vec4> tangents;
		std::span<const uint32_t> indices;
	};

	int get_num_faces( const SMikkTSpaceContext* context )
	{
		return static_cast<UserData*>( context->m_pUserData )->indices.size() / 3;
	}

	int get_num_vertices_of_face( const SMikkTSpaceContext*, const int )
	{
		// We only do triangles
		return 3;
	}

	void get_position( const SMikkTSpaceContext* context, float* pos_out, const int face, const int vert )
	{
		const auto data = static_cast<UserData*>( context->m_pUserData );
		const auto index = data->indices[ face * 3 + vert ];
		pos_out[ 0 ] = data->positions[ index ].x;
		pos_out[ 1 ] = data->positions[ index ].y;
		pos_out[ 2 ] = data->positions[ index ].z;
	}
	void get_normal( const SMikkTSpaceContext* context, float* normal_out, const int face, const int vert )
	{
		const auto data = static_cast<UserData*>( context->m_pUserData );
		const auto index = data->indices[ face * 3 + vert ];
		normal_out[ 0 ] = data->normals[ index ].x;
		normal_out[ 1 ] = data->normals[ index ].y;
		normal_out[ 2 ] = data->normals[ index ].z;
	}
	void get_texcoord( const SMikkTSpaceContext* context, float* uv_out, const int face, const int vert )
	{
		const auto data = static_cast<UserData*>( context->m_pUserData );
		const auto index = data->indices[ face * 3 + vert ];
		uv_out[ 0 ] = data->uvs[ index ].x;
		uv_out[ 1 ] = data->uvs[ index ].y;
	}

	void set_tangent( const SMikkTSpaceContext* context, const float* tangent, const float sign, const int face, const int vert )
	{
		const auto data = static_cast<UserData*>( context->m_pUserData );
		const auto index = data->indices[ face * 3 + vert ];
		data->tangents[ index ].x = tangent[ 0 ];
		data->tangents[ index ].y = tangent[ 1 ];
		data->tangents[ index ].z = tangent[ 2 ];
		data->tangents[ index ].w = sign;
	}

	static const SMikkTSpaceInterface interface { .m_getNumFaces = get_num_faces,
												  .m_getNumVerticesOfFace = get_num_vertices_of_face,
												  .m_getPosition = get_position,
												  .m_getNormal = get_normal,
												  .m_getTexCoord = get_texcoord,
												  .m_setTSpaceBasic = set_tangent };
}

class gpu_upload_job
{
	enum class state
	{
		UPLOADING,
		SUBMITTED,
		DONE
	};

public:
	gpu_upload_job( renderer::Device& device, uint32_t max_size )
		: _size( max_size )
		, _device( device )
		, _cmd_buffer( device.grab_command_buffer() )
		, _upload_buffer( device.create_buffer( renderer::Buffer::Usage::TRANSFER_SRC, _size, true ) )
		, _upload_fence( device.create_fence() )
	{
		_cmd_buffer->reset();
		_cmd_buffer->begin();
	}

	~gpu_upload_job() { assert( _state == state::DONE ); }

	void queue( const void* data, uint32_t size, renderer::Buffer& buffer, const uint32_t offset = 0 )
	{
		assert( _state == state::UPLOADING );
		assert( size < _size );
		if ( size + _offset > _size )
		{
			flush_full();
		}
		const auto ptr = static_cast<std::byte*>( _upload_buffer.get_mapped_address() );
		memcpy( ptr + _offset, data, size );
		_cmd_buffer->copy_buffer( _upload_buffer, _offset, size, buffer, offset );
		_offset += size;
	}

	template <typename T>
	void queue( std::span<const T> data, renderer::Buffer& buffer, const uint32_t offset = 0 )
	{
		assert( offset % alignof( T ) == 0 );
		queue( data.data(), data.size_bytes(), buffer, offset );
	}

	template <typename T>
	void queue( const std::vector<T>& data, renderer::Buffer& buffer, const uint32_t offset = 0 )
	{
		queue( std::span<const T>( data ), buffer, offset );
	}

	void queue( const SDL_Surface& surface, renderer::Texture& texture )
	{
		assert( _state == state::UPLOADING );
		const auto ptr = static_cast<std::byte*>( _upload_buffer.get_mapped_address() );
		const auto gpu_format = sdl_gpu_format( surface.format );
		const auto pitch = surface.w * get_format_size( sdl_to_vulkan_format( gpu_format ) );
		const auto size = pitch * surface.h;
		assert( size < _size );
		if ( size + _offset > _size )
		{
			flush_full();
		}
		if ( !SDL_ConvertPixels( surface.w, surface.h, surface.format, surface.pixels, surface.pitch, gpu_format, ptr + _offset, pitch ) )
		{
			throw renderer::Error( SDL_GetError() );
		}
		_cmd_buffer->transition_texture( texture, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::TRANSFER_DST_OPTIMAL );
		_cmd_buffer->copy_buffer_to_texture( _upload_buffer, _offset, texture );
		_cmd_buffer->transition_texture( texture,
										 renderer::Texture::Layout::TRANSFER_DST_OPTIMAL,
										 renderer::Texture::Layout::SHADER_READ_ONLY_OPTIMAL );
		_offset += size;
	}

	void submit()
	{
		assert( _state == state::UPLOADING );
		_cmd_buffer->end();
		_device.submit( *_cmd_buffer, _upload_fence );
		_state = state::SUBMITTED;
	}

	void wait_done()
	{
		assert( _state == state::SUBMITTED );
		_device.wait_for_fences( { _upload_fence }, 9999999999 );
		_state = state::DONE;
	}

	// Variadics are a headache and initializer list can't take references, pointers it is
	static void wait_all_done( std::initializer_list<gpu_upload_job*> jobs )
	{
		std::vector<renderer::Fence> fences;
		fences.reserve( jobs.size() );
		for ( const auto job : jobs )
		{
			fences.push_back( job->_upload_fence );
		}
		( *jobs.begin() )->_device.wait_for_fences( fences, 9999999999 );
		for ( const auto job : jobs )
		{
			job->_state = state::DONE;
		}
	}

private:
	void flush_full()
	{
		submit();
		// XXX: Could use 2 alternating buffers instead of one to avoid sync waiting
		wait_done();
		_device.reset_fences( { _upload_fence } );
		_offset = 0;
		_state = state::UPLOADING;
		_cmd_buffer->reset();
		_cmd_buffer->begin();
	}

	uint32_t _size;
	renderer::Device& _device;
	renderer::raii::CommandBuffer _cmd_buffer;
	renderer::raii::Buffer _upload_buffer;
	renderer::raii::Fence _upload_fence;
	uint32_t _offset = 0;
	state _state = state::UPLOADING;
};

class asset_load_job
{
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

	struct mesh_data
	{
		mesh_attribute positions;
		mesh_attribute normals;
		mesh_attribute tangents;
		mesh_attribute uvs;
		mesh_attribute indices;
		int albedo_idx = -1;
		int normal_idx = -1;
		int metallic_roughness_index = -1;
		pbr_data pbr_data;
	};


public:
	asset_load_job( renderer::Device& device, BindlessManager& bindless_manager, index_buffer& index_buffer, std::filesystem::path path )
		: _device( device )
		, _bindless_manager( bindless_manager )
		, _index_buffer( index_buffer )
		, _path( std::move( path ) )
		, _mesh_upload( device, 32 * 1024 * 1024 )
		, _texture_upload( device, 128 * 1024 * 1024 )
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
				case fastgltf::ComponentType::UnsignedShort:
				{
					switch ( accessor.type )
					{
						case fastgltf::AccessorType::Scalar:
							format = VK_FORMAT_R16_UINT;
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
			auto& m = _parsed_meshes.emplace_back();
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
				if ( material.pbrData.metallicRoughnessTexture )
				{
					_textures.push_back( make_texture( _asset, material.pbrData.metallicRoughnessTexture->textureIndex ) );
					m.metallic_roughness_index = _textures.size() - 1;
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
					if ( m.positions.format != VK_FORMAT_R32G32B32_SFLOAT )
					{
						throw renderer::Error( "Unsupported format for mesh positions" );
					}
				}
				else if ( attribute.name == "NORMAL" )
				{
					m.normals = make_attribute( _asset, attribute.accessorIndex );
					if ( m.normals.format != VK_FORMAT_R32G32B32_SFLOAT )
					{
						throw renderer::Error( "Unsupported format for mesh normals" );
					}
				}
				else if ( attribute.name == "TANGENT" )
				{
					m.tangents = make_attribute( _asset, attribute.accessorIndex );
					if ( m.tangents.format != VK_FORMAT_R32G32B32A32_SFLOAT )
					{
						throw renderer::Error( "Unsupported format for mesh tangents" );
					}
				}
				else if ( attribute.name == "TEXCOORD_0" )
				{
					m.uvs = make_attribute( _asset, attribute.accessorIndex );
					if ( m.uvs.format != VK_FORMAT_R32G32_SFLOAT )
					{
						throw renderer::Error( "Unsupported format for mesh texcoords" );
					}
				}
			}
			if ( mesh.primitives[ 0 ].indicesAccessor )
			{
				m.indices = make_attribute( _asset, *mesh.primitives[ 0 ].indicesAccessor );
				if ( m.indices.format != VK_FORMAT_R16_UINT && m.indices.format != VK_FORMAT_R32_UINT )
				{
					throw renderer::Error( "Unsupported format for mesh texcoords" );
				}
			}
		}

		_meshes.resize( _parsed_meshes.size() );
		_materials.resize( _parsed_meshes.size() );
	}

	void upload_meshes()
	{
		for ( int i = 0; i < _parsed_meshes.size(); ++i )
		{
			upload_mesh( i );
		}
		_mesh_upload.submit();
	}

	void upload_mesh( int mesh_index )
	{
		OPTICK_EVENT();
		const auto& gltf_buffer = std::get<fastgltf::sources::Array>( _asset.buffers[ 0 ].data ).bytes;
		const auto& parsed_mesh = _parsed_meshes[ mesh_index ];

		// Some models pack indices in u16 but we use u32 everywhere for now so we'll just convert them
		// It's only an issue for legacy rendering, meshlets do not care either way
		std::vector<uint32_t> expanded_indices;
		if ( parsed_mesh.indices.format == VK_FORMAT_R16_UINT )
		{
			expanded_indices.reserve( parsed_mesh.indices.length );
			const std::span indices( reinterpret_cast<const uint16_t*>( gltf_buffer.data() + parsed_mesh.indices.offset ),
									 parsed_mesh.indices.length );
			std::copy( begin( indices ), end( indices ), std::back_inserter( expanded_indices ) );
		}
		const std::span indices = expanded_indices.empty()
			? std::span( reinterpret_cast<const uint32_t*>( gltf_buffer.data() + parsed_mesh.indices.offset ), parsed_mesh.indices.length )
			: expanded_indices;

		const std::span positions( reinterpret_cast<const glm::vec3*>( gltf_buffer.data() + parsed_mesh.positions.offset ),
								   parsed_mesh.positions.length );
		const std::span normals( reinterpret_cast<const glm::vec3*>( gltf_buffer.data() + parsed_mesh.normals.offset ),
								 parsed_mesh.normals.length );
		const std::span uvs( reinterpret_cast<const glm::vec2*>( gltf_buffer.data() + parsed_mesh.uvs.offset ), parsed_mesh.uvs.length );

		std::vector<glm::vec4> generated_tangents;
		if ( !parsed_mesh.tangents.length )
		{
			OPTICK_EVENT( "generate_tangents" );
			generated_tangents.resize( positions.size() );
			MikkTSpace::UserData user_data { .positions = positions,
											 .normals = normals,
											 .uvs = uvs,
											 .tangents = generated_tangents,
											 .indices = indices };
			const SMikkTSpaceContext context { const_cast<SMikkTSpaceInterface*>( &MikkTSpace::interface ), &user_data };
			if ( !genTangSpaceDefault( &context ) )
			{
				throw renderer::Error( "Failed to generate tangets" );
			}
		}

		const auto tangents = parsed_mesh.tangents.length
			? std::span( reinterpret_cast<const glm::vec4*>( gltf_buffer.data() + parsed_mesh.tangents.offset ),
						 parsed_mesh.tangents.length )
			: generated_tangents;

		std::vector<mesh_vertex> vertex_data;
		{
			OPTICK_EVENT( "repack_vertices" );
			vertex_data.reserve( parsed_mesh.positions.length );

			for ( int v = 0; v < parsed_mesh.positions.length; ++v )
			{
				vertex_data.emplace_back( positions[ v ], uvs[ v ].x, normals[ v ], uvs[ v ].y, tangents[ v ] );
			}
		}

		std::vector<uint32_t> lod_indices;
		std::array<uint32_t, 3> indices_per_lod;
		{
			OPTICK_EVENT( "lod_generation" );
			constexpr std::array<float, 3> lod_indices_factor { { 0.2, 0.05, 0.01 } };
			constexpr std::array<float, 3> lod_error_factor { { 0.01, 0.04, 0.1 } };
			std::vector<uint32_t> buffer( indices.size() );
			for ( int lod = 0; lod < indices_per_lod.size(); ++lod )
			{
				const float attr_weights[ 3 ] = { 0.5f, 0.5f, 0.5f };
				const uint32_t size = meshopt_simplifyWithAttributes( buffer.data(),
																	  indices.data(),
																	  indices.size(),
																	  &vertex_data.front().position.x,
																	  vertex_data.size(),
																	  sizeof( mesh_vertex ),
																	  &vertex_data.front().normal.x,
																	  sizeof( mesh_vertex ),
																	  attr_weights,
																	  3,
																	  nullptr,
																	  parsed_mesh.indices.length * lod_indices_factor[ lod ],
																	  lod_error_factor[ lod ],
																	  0,
																	  nullptr );

				meshopt_optimizeVertexCache( buffer.data(), buffer.data(), size, vertex_data.size() );
				indices_per_lod[ lod ] = size;
				lod_indices.insert( end( lod_indices ), buffer.data(), buffer.data() + size );
			}
		}

		std::vector<gpu_meshlet> meshlets;
		std::vector<uint32_t> meshlet_vertices;
		std::vector<uint8_t> meshlet_triangles;
		std::vector<uint32_t> meshlets_per_lod;
		{
			OPTICK_EVENT( "meshlet_generation" );
			std::vector<meshopt_Meshlet> meshlet_buffer;
			uint32_t index_offset = 0;
			for ( int lod = 0; lod < 4; ++lod )
			{
				const auto indices_ptr = lod == 0 ? indices.data() : lod_indices.data() + index_offset;
				const auto indices_count = lod == 0 ? indices.size() : indices_per_lod[ lod - 1 ];
				index_offset += lod == 0 ? 0 : indices_count;

				auto max_meshlets = meshopt_buildMeshletsBound( indices_count, 64, 96 );
				meshlet_buffer.resize( max_meshlets );
				const auto vertices_offset = meshlet_vertices.size();
				meshlet_vertices.resize( meshlet_vertices.size() + indices_count );
				const auto triangles_offset = meshlet_triangles.size();
				meshlet_triangles.resize( meshlet_triangles.size() + indices_count + max_meshlets * 3 );

				const auto meshlet_count = meshopt_buildMeshlets( meshlet_buffer.data(),
																  meshlet_vertices.data() + vertices_offset,
																  meshlet_triangles.data() + triangles_offset,
																  indices_ptr,
																  indices_count,
																  &vertex_data.front().position.x,
																  vertex_data.size(),
																  sizeof( mesh_vertex ),
																  64,
																  96,
																  0.f );

				meshlets_per_lod.push_back( meshlet_count );
				const auto& last = meshlet_buffer[ meshlet_count - 1 ];

				meshlet_vertices.resize( vertices_offset + last.vertex_offset + last.vertex_count );
				meshlet_triangles.resize( triangles_offset + last.triangle_offset + last.triangle_count * 3 );

				for ( const auto& meshlet : std::span( meshlet_buffer.data(), meshlet_count ) )
				{
					const uint32_t vertex_index = meshlet.vertex_offset + vertices_offset;
					const uint32_t triangle_index = meshlet.triangle_offset + triangles_offset;
					const auto bounds = meshopt_computeMeshletBounds( meshlet_vertices.data() + vertex_index,
																	  meshlet_triangles.data() + triangle_index,
																	  meshlet.triangle_count,
																	  &vertex_data.front().position.x,
																	  vertex_data.size(),
																	  sizeof( mesh_vertex ) );

					meshlets.emplace_back( vertex_index,
										   triangle_index,
										   meshlet.vertex_count,
										   meshlet.triangle_count,
										   glm::aligned_vec4( bounds.center[ 0 ], bounds.center[ 1 ], bounds.center[ 2 ], bounds.radius ) );
				}
			}
		}

		auto& mesh = _meshes[ mesh_index ];

		const auto bounds = meshopt_computeSphereBounds( &vertex_data[ 0 ].position.x,
														 vertex_data.size(),
														 sizeof( mesh_vertex ),
														 nullptr,
														 0 );
		mesh.gpu_data.bounding_sphere.x = bounds.center[ 0 ];
		mesh.gpu_data.bounding_sphere.y = bounds.center[ 1 ];
		mesh.gpu_data.bounding_sphere.z = bounds.center[ 2 ];
		mesh.gpu_data.bounding_sphere.w = bounds.radius;

		// Index buffer
		uint32_t index_offset = _index_buffer.size.fetch_add( indices.size() );
		_mesh_upload.queue( indices, _index_buffer.buffer, index_offset * sizeof( uint32_t ) );
		mesh.gpu_data.lods[ 0 ].indices_offset = index_offset;
		mesh.gpu_data.lods[ 0 ].indices_count = parsed_mesh.indices.length;

		index_offset = _index_buffer.size.fetch_add( lod_indices.size() );
		_mesh_upload.queue( lod_indices, _index_buffer.buffer, index_offset * sizeof( uint32_t ) );
		for ( int lod = 1; lod < mesh.gpu_data.lods.size(); ++lod )
		{
			mesh.gpu_data.lods[ lod ].indices_offset = mesh.gpu_data.lods[ lod - 1 ].indices_offset
				+ mesh.gpu_data.lods[ lod - 1 ].indices_count;
			mesh.gpu_data.lods[ lod ].indices_count = indices_per_lod[ lod - 1 ];
		}

		// Data buffer (vertices + meshlets)
		auto data_buffer_size = vertex_data.size() * sizeof( mesh_vertex );
		data_buffer_size += meshlets.size() * sizeof( gpu_meshlet );
		data_buffer_size += meshlet_vertices.size() * sizeof( uint32_t );
		data_buffer_size += meshlet_triangles.size() * sizeof( uint8_t );

		mesh.data_buffer = _device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
													  | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
												  data_buffer_size );

		mesh.gpu_data.vertices = mesh.data_buffer.get_device_address();
		_mesh_upload.queue( vertex_data, mesh.data_buffer );
		auto offset = vertex_data.size() * sizeof( mesh_vertex );

		mesh.gpu_data.meshlets = mesh.data_buffer.get_device_address() + offset;
		_mesh_upload.queue( meshlets, mesh.data_buffer, offset );
		offset += meshlets.size() * sizeof( gpu_meshlet );

		for ( int lod = 0; lod < 4; ++lod )
		{
			mesh.gpu_data.lods[ lod ].meshlets_count = meshlets_per_lod[ lod ];
			if ( lod > 0 )
			{
				mesh.gpu_data.lods[ lod ].meshlets_offset = mesh.gpu_data.lods[ lod - 1 ].meshlets_offset
					+ mesh.gpu_data.lods[ lod - 1 ].meshlets_count;
			}
		}

		mesh.gpu_data.meshlet_vertices = mesh.data_buffer.get_device_address() + offset;
		_mesh_upload.queue( meshlet_vertices, mesh.data_buffer, offset );
		offset += meshlet_vertices.size() * sizeof( uint32_t );

		mesh.gpu_data.meshlet_triangles = mesh.data_buffer.get_device_address() + offset;
		_mesh_upload.queue( meshlet_triangles, mesh.data_buffer, offset );
		offset += meshlet_triangles.size() * sizeof( uint8_t );

		mesh.handle = _bindless_manager.add_buffer_entry( mesh.gpu_data );
	}

	void decompress_textures()
	{
		OPTICK_EVENT();
		const auto& gltf_buffer = std::get<fastgltf::sources::Array>( _asset.buffers[ 0 ].data ).bytes;

		tbb::parallel_for(
			0uz,
			_textures.size(),
			[ & ]( size_t index )
			{
				OPTICK_EVENT( "decompress_tex" );
				_textures[ index ].image.reset(
					IMG_Load_IO( SDL_IOFromConstMem( gltf_buffer.data() + _textures[ index ].offset, _textures[ index ].size ), true ) );
				if ( !_textures[ index ].image )
				{
					throw renderer::Error( SDL_GetError() );
				}
				else if ( _textures[ index ].image->format == SDL_PIXELFORMAT_INDEX8 )
				{
					sdl::raii::Surface converted_image( SDL_ConvertSurface( _textures[ index ].image.get(), SDL_PIXELFORMAT_RGBA32 ) );
					if ( !converted_image )
					{
						throw renderer::Error( SDL_GetError() );
					}
					_textures[ index ].image = std::move( converted_image );
				}
			} );
	}

	void upload_materials()
	{
		OPTICK_EVENT();

		std::size_t alloc_size = 0;
		for ( auto& tex : _textures )
		{
			alloc_size += tex.image->w * tex.image->h * get_format_size( sdl_to_vulkan_format( sdl_gpu_format( tex.image->format ) ) );
		}

		std::size_t offset = 0;
		for ( int i = 0; i < _meshes.size(); ++i )
		{
			const auto& parsed_mesh = _parsed_meshes[ i ];
			auto& material = _materials[ i ];

			auto tex = make_image_from_surface( _device, *_textures[ parsed_mesh.albedo_idx ].image, true );
			_texture_upload.queue( *_textures[ parsed_mesh.albedo_idx ].image, tex );
			material.gpu_data.albedo_texture_index = _bindless_manager.add_texture( std::move( tex ) ).handles.texture_index;

			if ( parsed_mesh.normal_idx != -1 )
			{
				tex = make_image_from_surface( _device, *_textures[ parsed_mesh.normal_idx ].image, false );
				_texture_upload.queue( *_textures[ parsed_mesh.normal_idx ].image, tex );
				material.gpu_data.normal_texture_index = _bindless_manager.add_texture( std::move( tex ) ).handles.texture_index;
			}

			if ( parsed_mesh.metallic_roughness_index != -1 )
			{
				tex = make_image_from_surface( _device, *_textures[ parsed_mesh.metallic_roughness_index ].image, false );
				_texture_upload.queue( *_textures[ parsed_mesh.metallic_roughness_index ].image, tex );
				material.gpu_data.metallic_roughness_texture_index = _bindless_manager.add_texture( std::move( tex ) )
																		 .handles.texture_index;
			}

			material.gpu_data.pbr_data = parsed_mesh.pbr_data;

			material.handle = _bindless_manager.add_buffer_entry( material.gpu_data );
		}

		_texture_upload.submit();
	}

	asset finalize()
	{
		OPTICK_EVENT();
		std::vector<glm::vec4> bounding_spheres;
		bounding_spheres.reserve( _meshes.size() );
		for ( const auto& mesh : _meshes )
		{
			bounding_spheres.push_back( mesh.gpu_data.bounding_sphere );
		}
		const auto bounds = meshopt_computeSphereBounds( &bounding_spheres[ 0 ].x,
														 bounding_spheres.size(),
														 sizeof( glm::vec4 ),
														 &bounding_spheres[ 0 ].w,
														 sizeof( glm::vec4 ) );

		gpu_upload_job::wait_all_done( { &_mesh_upload, &_texture_upload } );
		return asset { std::move( _meshes ),
					   std::move( _materials ),
					   glm::vec4( bounds.center[ 0 ], bounds.center[ 1 ], bounds.center[ 2 ], bounds.radius ) };
	}

	std::size_t get_texture_count() const { return _textures.size(); }

private:
	renderer::Device& _device;
	BindlessManager& _bindless_manager;
	index_buffer& _index_buffer;
	std::filesystem::path _path;
	gpu_upload_job _mesh_upload;
	gpu_upload_job _texture_upload;
	fastgltf::Asset _asset;
	std::vector<mesh_texture> _textures;
	std::vector<mesh_data> _parsed_meshes;
	std::vector<mesh> _meshes;
	std::vector<material> _materials;
};

asset load_asset( renderer::Device& device, BindlessManager& bindless_manager, index_buffer& index_buffer, std::filesystem::path path )
{
	OPTICK_EVENT();
	asset_load_job job( device, bindless_manager, index_buffer, std::move( path ) );
	job.parse();
	tbb::parallel_invoke( [ & ]() { job.upload_meshes(); },
						  [ & ]
						  {
							  job.decompress_textures();
							  job.upload_materials();
						  } );
	return job.finalize();
}
