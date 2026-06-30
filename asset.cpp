#define GLM_ENABLE_EXPERIMENTAL

#include "asset.h"

#include <SDL3_image/SDL_image.h>
#include <fastgltf/core.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <istream>
#include <meshoptimizer.h>
#include <mikktspace.h>
#include <numeric>
#include <renderer/command_buffer.h>
#include <renderer/details/profiler.h>
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


namespace
{
	template <class... Ts>
	struct overloads : Ts...
	{
		using Ts::operator()...;
	};

	template <typename T>
	class attribute_accessor
	{
	public:
		attribute_accessor( const std::byte* buffer, size_t count, uint32_t stride )
			: _stride( stride != 0 ? stride : sizeof( T ) )
			, _data( buffer, count * _stride )
		{
		}

		explicit attribute_accessor( std::span<const T> data )
			: _stride( sizeof( T ) )
			, _data( reinterpret_cast<const std::byte*>( data.data() ), data.size_bytes() )
		{
		}

		explicit attribute_accessor( const std::vector<T> data )
			: attribute_accessor( std::span( data ) )
		{
		}

		const T& get( int index ) const { return reinterpret_cast<const T&>( _data[ index * _stride ] ); }
		size_t size() const { return _data.size(); }

	private:
		uint32_t _stride;
		std::span<const std::byte> _data;
	};
}

namespace MikkTSpace
{
	struct UserData
	{
		attribute_accessor<glm::vec3> positions;
		attribute_accessor<glm::vec3> normals;
		attribute_accessor<glm::vec2> uvs;
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
		const auto& position = data->positions.get( index );
		pos_out[ 0 ] = position.x;
		pos_out[ 1 ] = position.y;
		pos_out[ 2 ] = position.z;
	}
	void get_normal( const SMikkTSpaceContext* context, float* normal_out, const int face, const int vert )
	{
		const auto data = static_cast<UserData*>( context->m_pUserData );
		const auto index = data->indices[ face * 3 + vert ];
		const auto& normal = data->normals.get( index );
		normal_out[ 0 ] = normal.x;
		normal_out[ 1 ] = normal.y;
		normal_out[ 2 ] = normal.z;
	}
	void get_texcoord( const SMikkTSpaceContext* context, float* uv_out, const int face, const int vert )
	{
		const auto data = static_cast<UserData*>( context->m_pUserData );
		const auto index = data->indices[ face * 3 + vert ];
		const auto& uv = data->uvs.get( index );
		uv_out[ 0 ] = uv.x;
		uv_out[ 1 ] = uv.y;
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
		assert( offset + size <= buffer.get_size() );
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
		uint32_t buffer_index = -1;
	};

	struct mesh_data
	{
		mesh_attribute positions;
		mesh_attribute normals;
		mesh_attribute tangents;
		mesh_attribute uvs;
		mesh_attribute indices;
	};

	struct mesh_upload_data
	{
		std::vector<mesh_vertex> vertices;
		std::span<const uint32_t> indices;
		std::vector<uint32_t> lod_indices;
		std::array<uint32_t, 3> indices_per_lod;
		std::vector<uint32_t> expanded_indices;
		std::vector<gpu_meshlet> meshlets;
		std::vector<uint32_t> meshlet_vertices;
		std::vector<uint8_t> meshlet_triangles;
		std::vector<uint32_t> meshlets_per_lod;
	};

	enum class texture_usage
	{
		UNUSED,
		LINEAR,
		SRGB
	};

	struct data_source
	{
		std::span<const std::byte> data;
		std::unique_ptr<std::byte[]> buffer;
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

	asset run()
	{
		OPTICK_EVENT();
		parse();
		tbb::parallel_invoke(
			[ & ]()
			{
				prepare_meshes();
				upload_meshes();
			},
			[ & ]
			{
				decompress_textures();
				upload_textures();
			} );
		return finalize();
	}

private:
	void add_meshes( size_t node_index )
	{
		if ( _asset.nodes[ node_index ].meshIndex )
		{
			_meshes_usage[ *_asset.nodes[ node_index ].meshIndex ] = 1;
		}
		for ( const auto node : _asset.nodes[ node_index ].children )
		{
			add_meshes( node );
		}
	}

	void add_texture_usage( size_t index, bool sRGB )
	{
		const auto usage = sRGB ? texture_usage::SRGB : texture_usage::LINEAR;
		if ( _textures_usage[ index ] == texture_usage::UNUSED )
		{
			const auto image_index = *_asset.textures[ index ].imageIndex;
			if ( _images_usage[ image_index ] )
			{
				throw renderer::Error( "Unsupported multiple texture references to the same image" );
			}
			_images_usage[ image_index ] = 1;
			_textures_usage[ index ] = usage;
		}
		else if ( _textures_usage[ index ] != usage )
		{
			throw renderer::Error( "Unsupported mixed use of texture as both LINEAR and SRGB" );
		}
	}

	mesh_attribute parse_attribute( size_t accessor_index ) const
	{
		const auto& accessor = _asset.accessors[ accessor_index ];
		const auto& view = _asset.bufferViews[ *accessor.bufferViewIndex ];
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
								.format = format,
								.buffer_index = static_cast<uint32_t>( view.bufferIndex ) };
	};

	data_source load( const fastgltf::DataSource& source, bool allow_bufferview ) const
	{
		return std::visit(
			overloads( [ this ]( const fastgltf::sources::Array& array )
					   { return data_source { .data = { array.bytes.data(), array.bytes.size() } }; },
					   [ this, allow_bufferview ]( const fastgltf::sources::BufferView& view )
					   {
						   if ( !allow_bufferview )
						   {
							   throw renderer::Error( "Unsupported recursive buffer view" );
						   }
						   const auto& buffer_view = _asset.bufferViews[ view.bufferViewIndex ];
						   return data_source { .data = _buffers[ buffer_view.bufferIndex ].data.subspan( buffer_view.byteOffset,
																										  buffer_view.byteLength ) };
					   },
					   [ this ]( const fastgltf::sources::URI& uri )
					   {
						   if ( !uri.uri.isLocalPath() )
						   {
							   throw renderer::Error( "Unsupported remote data source" );
						   }
						   else if ( uri.fileByteOffset != 0 )
						   {
							   throw renderer::Error( "Unsupported local data source with offset (need to implement views into files)" );
						   }
						   else
						   {
							   auto path( uri.uri.fspath() );
							   if ( !path.is_absolute() )
								   path = _path.parent_path() / path;
							   std::ifstream istream( path, std::ios::binary | std::ios::ate );
							   if ( !istream )
							   {
								   throw renderer::Error( "Asset references non existing file" );
							   }
							   const size_t size = istream.tellg();
							   auto buffer = std::make_unique<std::byte[]>( size );
							   istream.seekg( 0 );
							   istream.read( reinterpret_cast<char*>( buffer.get() ), size );
							   return data_source { .data = { buffer.get(), size }, .buffer = std::move( buffer ) };
						   }
					   },
					   []( const auto& ) -> data_source { throw renderer::Error( "Unsupported data source" ); } ),
			source );
	}

	void parse()
	{
		OPTICK_EVENT();

		fastgltf::Parser parser( fastgltf::Extensions::KHR_materials_specular );
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

		_meshes_usage.resize( _asset.meshes.size() );
		for ( const auto& scene : _asset.scenes )
		{
			for ( const auto& node : scene.nodeIndices )
			{
				add_meshes( node );
			}
		}

		_materials_usage.resize( _asset.materials.size(), 0 );

		for ( int i = 0; i < _meshes_usage.size(); ++i )
		{
			if ( _meshes_usage[ i ] )
			{
				auto& mesh = _asset.meshes[ i ];
				if ( mesh.primitives.size() != 1 )
				{
					throw renderer::Error( "Unsupported mesh with multiple primitives" );
				}
				if ( mesh.primitives[ 0 ].materialIndex )
				{
					_materials_usage[ *mesh.primitives[ 0 ].materialIndex ] = 1;
				}
				auto& m = _parsed_meshes.emplace_back();

				for ( const auto& attribute : mesh.primitives[ 0 ].attributes )
				{
					if ( attribute.name == "POSITION" )
					{
						m.positions = parse_attribute( attribute.accessorIndex );
						if ( m.positions.format != VK_FORMAT_R32G32B32_SFLOAT )
						{
							throw renderer::Error( "Unsupported format for mesh positions" );
						}
					}
					else if ( attribute.name == "NORMAL" )
					{
						m.normals = parse_attribute( attribute.accessorIndex );
						if ( m.normals.format != VK_FORMAT_R32G32B32_SFLOAT )
						{
							throw renderer::Error( "Unsupported format for mesh normals" );
						}
					}
					else if ( attribute.name == "TANGENT" )
					{
						m.tangents = parse_attribute( attribute.accessorIndex );
						if ( m.tangents.format != VK_FORMAT_R32G32B32A32_SFLOAT )
						{
							throw renderer::Error( "Unsupported format for mesh tangents" );
						}
					}
					else if ( attribute.name == "TEXCOORD_0" )
					{
						m.uvs = parse_attribute( attribute.accessorIndex );
						if ( m.uvs.format != VK_FORMAT_R32G32_SFLOAT )
						{
							throw renderer::Error( "Unsupported format for mesh texcoords" );
						}
					}
				}
				if ( mesh.primitives[ 0 ].indicesAccessor )
				{
					m.indices = parse_attribute( *mesh.primitives[ 0 ].indicesAccessor );
					if ( m.indices.format != VK_FORMAT_R16_UINT && m.indices.format != VK_FORMAT_R32_UINT )
					{
						throw renderer::Error( "Unsupported format for mesh texcoords" );
					}
				}
			}
		}

		_images_usage.resize( _asset.images.size(), 0 );
		_textures_usage.resize( _asset.textures.size(), texture_usage::UNUSED );
		for ( int i = 0; i < _materials_usage.size(); ++i )
		{
			if ( _materials_usage[ i ] )
			{
				const auto& material = _asset.materials[ i ];
				if ( material.pbrData.baseColorTexture )
				{
					add_texture_usage( material.pbrData.baseColorTexture->textureIndex, true );
				}
				if ( material.normalTexture )
				{
					add_texture_usage( material.normalTexture->textureIndex, false );
				}
				if ( material.pbrData.metallicRoughnessTexture )
				{
					add_texture_usage( material.pbrData.metallicRoughnessTexture->textureIndex, false );
				}
				if ( material.specular )
				{
					if ( material.specular->specularTexture )
					{
						add_texture_usage( material.specular->specularTexture->textureIndex, false );
					}
				}
				// We ignore emissive for now
			}
		}

		// For now we assume all buffers are used one way or the other
		// If that's wrong we'll potentially do useless file I/O
		_buffers.reserve( _asset.buffers.size() );
		for ( const auto& buffer : _asset.buffers )
		{
			_buffers.emplace_back( load( buffer.data, false ) );
		}
	}

	void prepare_meshes()
	{
		OPTICK_EVENT();
		_mesh_upload_data.resize( _parsed_meshes.size() );
		tbb::parallel_for( 0zu, _parsed_meshes.size(), [ this ]( size_t index ) { prepare_mesh( index ); } );
#if 0
		for ( int i = 0; i < _parsed_meshes.size(); ++i )
		{
			prepare_mesh( i );
		}
#endif
	}

	void prepare_mesh( uint32_t mesh_index )
	{
		OPTICK_EVENT();
		const auto& parsed_mesh = _parsed_meshes[ mesh_index ];
		auto& upload_data = _mesh_upload_data[ mesh_index ];

		// Some models pack indices in u16 but we use u32 everywhere for now so we'll just convert them
		// It's only an issue for legacy rendering, meshlets do not care either way
		if ( parsed_mesh.indices.format == VK_FORMAT_R16_UINT )
		{
			upload_data.expanded_indices.reserve( parsed_mesh.indices.length );
			const std::span indices(
				reinterpret_cast<const uint16_t*>( _buffers[ parsed_mesh.indices.buffer_index ].data.data() + parsed_mesh.indices.offset ),
				parsed_mesh.indices.length );
			std::copy( begin( indices ), end( indices ), std::back_inserter( upload_data.expanded_indices ) );
		}
		upload_data.indices = upload_data.expanded_indices.empty()
			? std::span( reinterpret_cast<const uint32_t*>( _buffers[ parsed_mesh.indices.buffer_index ].data.data()
															+ parsed_mesh.indices.offset ),
						 parsed_mesh.indices.length )
			: upload_data.expanded_indices;

		const attribute_accessor<glm::vec3> positions( _buffers[ parsed_mesh.positions.buffer_index ].data.data()
														   + parsed_mesh.positions.offset,
													   parsed_mesh.positions.length,
													   parsed_mesh.positions.stride );
		const attribute_accessor<glm::vec3> normals( _buffers[ parsed_mesh.normals.buffer_index ].data.data() + parsed_mesh.normals.offset,
													 parsed_mesh.normals.length,
													 parsed_mesh.normals.stride );
		const attribute_accessor<glm::vec2> uvs( _buffers[ parsed_mesh.uvs.buffer_index ].data.data() + parsed_mesh.uvs.offset,
												 parsed_mesh.uvs.length,
												 parsed_mesh.uvs.stride );

		std::vector<glm::vec4> generated_tangents;
		auto tangents_stride = parsed_mesh.tangents.stride;
		if ( !parsed_mesh.tangents.length )
		{
			OPTICK_EVENT( "generate_tangents" );
			generated_tangents.resize( positions.size() );
			MikkTSpace::UserData user_data { .positions = positions,
											 .normals = normals,
											 .uvs = uvs,
											 .tangents = generated_tangents,
											 .indices = upload_data.indices };
			const SMikkTSpaceContext context { const_cast<SMikkTSpaceInterface*>( &MikkTSpace::interface ), &user_data };
			if ( !genTangSpaceDefault( &context ) )
			{
				throw renderer::Error( "Failed to generate tangets" );
			}
			tangents_stride = sizeof( glm::vec4 );
		}

		const attribute_accessor<glm::vec4> tangents = generated_tangents.size()
			? attribute_accessor<glm::vec4>( generated_tangents )
			: attribute_accessor<glm::vec4>( _buffers[ parsed_mesh.tangents.buffer_index ].data.data() + parsed_mesh.tangents.offset,
											 parsed_mesh.tangents.length,
											 tangents_stride );

		{
			OPTICK_EVENT( "repack_vertices" );
			upload_data.vertices.reserve( parsed_mesh.positions.length );

			for ( int v = 0; v < parsed_mesh.positions.length; ++v )
			{
				const auto& uv = uvs.get( v );
				upload_data.vertices.emplace_back( positions.get( v ), uv.x, normals.get( v ), uv.y, tangents.get( v ) );
			}
		}

		{
			OPTICK_EVENT( "lod_generation" );
			constexpr std::array<float, 3> lod_indices_factor { { 0.2, 0.05, 0.01 } };
			constexpr std::array<float, 3> lod_error_factor { { 0.01, 0.04, 0.1 } };
			std::vector<uint32_t> buffer( upload_data.indices.size() );
			for ( int lod = 0; lod < upload_data.indices_per_lod.size(); ++lod )
			{
				const float attr_weights[ 3 ] = { 0.5f, 0.5f, 0.5f };
				const uint32_t size = meshopt_simplifyWithAttributes( buffer.data(),
																	  upload_data.indices.data(),
																	  upload_data.indices.size(),
																	  &upload_data.vertices.front().position.x,
																	  upload_data.vertices.size(),
																	  sizeof( mesh_vertex ),
																	  &upload_data.vertices.front().normal.x,
																	  sizeof( mesh_vertex ),
																	  attr_weights,
																	  3,
																	  nullptr,
																	  parsed_mesh.indices.length * lod_indices_factor[ lod ],
																	  lod_error_factor[ lod ],
																	  0,
																	  nullptr );

				meshopt_optimizeVertexCache( buffer.data(), buffer.data(), size, upload_data.vertices.size() );
				upload_data.indices_per_lod[ lod ] = size;
				upload_data.lod_indices.insert( end( upload_data.lod_indices ), buffer.data(), buffer.data() + size );
			}
		}

		{
			OPTICK_EVENT( "meshlet_generation" );
			std::vector<meshopt_Meshlet> meshlet_buffer;
			uint32_t index_offset = 0;
			for ( int lod = 0; lod < 4; ++lod )
			{
				const auto indices_ptr = lod == 0 ? upload_data.indices.data() : upload_data.lod_indices.data() + index_offset;
				const auto indices_count = lod == 0 ? upload_data.indices.size() : upload_data.indices_per_lod[ lod - 1 ];
				index_offset += lod == 0 ? 0 : indices_count;

				if ( indices_count > 0 )
				{
					auto max_meshlets = meshopt_buildMeshletsBound( indices_count, 64, 96 );
					meshlet_buffer.resize( max_meshlets );
					const auto vertices_offset = upload_data.meshlet_vertices.size();
					upload_data.meshlet_vertices.resize( upload_data.meshlet_vertices.size() + indices_count );
					const auto triangles_offset = upload_data.meshlet_triangles.size();
					upload_data.meshlet_triangles.resize( upload_data.meshlet_triangles.size() + indices_count + max_meshlets * 3 );

					const auto meshlet_count = meshopt_buildMeshlets( meshlet_buffer.data(),
																	  upload_data.meshlet_vertices.data() + vertices_offset,
																	  upload_data.meshlet_triangles.data() + triangles_offset,
																	  indices_ptr,
																	  indices_count,
																	  &upload_data.vertices.front().position.x,
																	  upload_data.vertices.size(),
																	  sizeof( mesh_vertex ),
																	  64,
																	  96,
																	  0.f );

					upload_data.meshlets_per_lod.push_back( meshlet_count );
					const auto& last = meshlet_buffer[ meshlet_count - 1 ];

					upload_data.meshlet_vertices.resize( vertices_offset + last.vertex_offset + last.vertex_count );
					upload_data.meshlet_triangles.resize( triangles_offset + last.triangle_offset + last.triangle_count * 3 );

					for ( const auto& meshlet : std::span( meshlet_buffer.data(), meshlet_count ) )
					{
						const uint32_t vertex_index = meshlet.vertex_offset + vertices_offset;
						const uint32_t triangle_index = meshlet.triangle_offset + triangles_offset;
						const auto bounds = meshopt_computeMeshletBounds( upload_data.meshlet_vertices.data() + vertex_index,
																		  upload_data.meshlet_triangles.data() + triangle_index,
																		  meshlet.triangle_count,
																		  &upload_data.vertices.front().position.x,
																		  upload_data.vertices.size(),
																		  sizeof( mesh_vertex ) );

						upload_data.meshlets.emplace_back(
							vertex_index,
							triangle_index,
							meshlet.vertex_count,
							meshlet.triangle_count,
							glm::vec4( bounds.center[ 0 ], bounds.center[ 1 ], bounds.center[ 2 ], bounds.radius ),
							glm::vec4( bounds.cone_axis[ 0 ], bounds.cone_axis[ 1 ], bounds.cone_axis[ 2 ], bounds.cone_cutoff ) );
					}
				}
				else
				{
					upload_data.meshlets_per_lod.push_back( 0 );
				}
			}
		}
	}

	void upload_meshes()
	{
		OPTICK_EVENT();
		_meshes.resize( _parsed_meshes.size() );
		for ( int i = 0; i < _parsed_meshes.size(); ++i )
		{
			upload_mesh( i );
		}
		_mesh_upload.submit();
	}

	void upload_mesh( int mesh_index )
	{
		OPTICK_EVENT();
		const auto& upload_data = _mesh_upload_data[ mesh_index ];
		auto& mesh = _meshes[ mesh_index ];

		const auto bounds = meshopt_computeSphereBounds( &upload_data.vertices[ 0 ].position.x,
														 upload_data.vertices.size(),
														 sizeof( mesh_vertex ),
														 nullptr,
														 0 );
		mesh.gpu_data.bounding_sphere.x = bounds.center[ 0 ];
		mesh.gpu_data.bounding_sphere.y = bounds.center[ 1 ];
		mesh.gpu_data.bounding_sphere.z = bounds.center[ 2 ];
		mesh.gpu_data.bounding_sphere.w = bounds.radius;

		// Index buffer
		uint32_t index_offset = _index_buffer.size.fetch_add( upload_data.indices.size() );
		_mesh_upload.queue( upload_data.indices, _index_buffer.buffer, index_offset * sizeof( uint32_t ) );
		mesh.gpu_data.lods[ 0 ].indices_offset = index_offset;
		mesh.gpu_data.lods[ 0 ].indices_count = upload_data.indices.size();

		index_offset = _index_buffer.size.fetch_add( upload_data.lod_indices.size() );
		_mesh_upload.queue( upload_data.lod_indices, _index_buffer.buffer, index_offset * sizeof( uint32_t ) );
		for ( int lod = 1; lod < mesh.gpu_data.lods.size(); ++lod )
		{
			mesh.gpu_data.lods[ lod ].indices_offset = mesh.gpu_data.lods[ lod - 1 ].indices_offset
				+ mesh.gpu_data.lods[ lod - 1 ].indices_count;
			mesh.gpu_data.lods[ lod ].indices_count = upload_data.indices_per_lod[ lod - 1 ];
		}

		// Data buffer (vertices + meshlets)
		auto data_buffer_size = upload_data.vertices.size() * sizeof( mesh_vertex );
		data_buffer_size += upload_data.meshlets.size() * sizeof( gpu_meshlet );
		data_buffer_size += upload_data.meshlet_vertices.size() * sizeof( uint32_t );
		data_buffer_size += upload_data.meshlet_triangles.size() * sizeof( uint8_t );

		mesh.data_buffer = _device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
													  | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
												  data_buffer_size );

		mesh.gpu_data.vertices = mesh.data_buffer.get_device_address();
		_mesh_upload.queue( upload_data.vertices, mesh.data_buffer );
		auto offset = upload_data.vertices.size() * sizeof( mesh_vertex );

		mesh.gpu_data.meshlets = mesh.data_buffer.get_device_address() + offset;
		_mesh_upload.queue( upload_data.meshlets, mesh.data_buffer, offset );
		offset += upload_data.meshlets.size() * sizeof( gpu_meshlet );

		for ( int lod = 0; lod < 4; ++lod )
		{
			mesh.gpu_data.lods[ lod ].meshlets_count = upload_data.meshlets_per_lod[ lod ];
			if ( lod > 0 )
			{
				mesh.gpu_data.lods[ lod ].meshlets_offset = mesh.gpu_data.lods[ lod - 1 ].meshlets_offset
					+ mesh.gpu_data.lods[ lod - 1 ].meshlets_count;
			}
		}

		mesh.gpu_data.meshlet_vertices = mesh.data_buffer.get_device_address() + offset;
		_mesh_upload.queue( upload_data.meshlet_vertices, mesh.data_buffer, offset );
		offset += upload_data.meshlet_vertices.size() * sizeof( uint32_t );

		mesh.gpu_data.meshlet_triangles = mesh.data_buffer.get_device_address() + offset;
		_mesh_upload.queue( upload_data.meshlet_triangles, mesh.data_buffer, offset );
		offset += upload_data.meshlet_triangles.size() * sizeof( uint8_t );

		mesh.handle = _bindless_manager.add_buffer_entry( mesh.gpu_data );
		mesh.material_index = *_asset.meshes[ mesh_index ].primitives[ 0 ].materialIndex;
	}

	void decompress_textures()
	{
		OPTICK_EVENT();
		_images.resize( _images_usage.size() );

		tbb::parallel_for(
			0uz,
			_images_usage.size(),
			[ & ]( size_t index )
			{
				OPTICK_EVENT( "decompress_tex" );
				if ( _images_usage[ index ] )
				{
					const auto source = load( _asset.images[ index ].data, true );
					_images[ index ].reset( IMG_Load_IO( SDL_IOFromConstMem( source.data.data(), source.data.size() ), true ) );
					if ( !_images[ index ] )
					{
						throw renderer::Error( SDL_GetError() );
					}
					else if ( _images[ index ]->format == SDL_PIXELFORMAT_INDEX8 )
					{
						sdl::raii::Surface converted_image( SDL_ConvertSurface( _images[ index ].get(), SDL_PIXELFORMAT_RGBA32 ) );
						if ( !converted_image )
						{
							throw renderer::Error( SDL_GetError() );
						}
						_images[ index ] = std::move( converted_image );
					}
				}
			} );
	}

	void upload_textures()
	{
		OPTICK_EVENT();
		_textures.resize( _textures_usage.size() );
		std::size_t offset = 0;
		for ( int i = 0; i < _textures_usage.size(); ++i )
		{
			if ( _textures_usage[ i ] != texture_usage::UNUSED )
			{
				const auto image_index = *_asset.textures[ i ].imageIndex;
				auto tex = make_image_from_surface( _device, *_images[ image_index ], _textures_usage[ i ] == texture_usage::SRGB );
				_texture_upload.queue( *_images[ image_index ], tex );
				_textures[ i ] = _bindless_manager.add_texture( std::move( tex ) );
			}
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

		_materials.resize( _asset.materials.size() );
		for ( int i = 0; i < _materials_usage.size(); ++i )
		{
			if ( _materials_usage[ i ] )
			{
				auto& material = _materials[ i ];
				if ( _asset.materials[ i ].pbrData.baseColorTexture )
				{
					material.gpu_data.albedo_texture_index = _textures[ _asset.materials[ i ].pbrData.baseColorTexture->textureIndex ]
																 .handles.texture_index;
				}

				if ( _asset.materials[ i ].normalTexture )
				{
					material.gpu_data.normal_texture_index = _textures[ _asset.materials[ i ].normalTexture->textureIndex ]
																 .handles.texture_index;
				}

				if ( _asset.materials[ i ].pbrData.metallicRoughnessTexture )
				{
					material.gpu_data.metallic_roughness_texture_index = _textures[ _asset.materials[ i ]
																						.pbrData.metallicRoughnessTexture->textureIndex ]
																			 .handles.texture_index;
				}
				if ( _asset.materials[ i ].specular )
				{
					if ( _asset.materials[ i ].specular->specularTexture )
					{
						material.gpu_data.specular_texture_index = _textures[ _asset.materials[ i ]
																				  .specular->specularTexture->textureIndex ]
																	   .handles.texture_index;
					}
				}
				material.gpu_data.pbr_data.base_color.x = _asset.materials[ i ].pbrData.baseColorFactor.x();
				material.gpu_data.pbr_data.base_color.y = _asset.materials[ i ].pbrData.baseColorFactor.y();
				material.gpu_data.pbr_data.base_color.z = _asset.materials[ i ].pbrData.baseColorFactor.z();
				material.gpu_data.pbr_data.base_color.w = _asset.materials[ i ].pbrData.baseColorFactor.w();
				material.gpu_data.pbr_data.metalness = _asset.materials[ i ].pbrData.metallicFactor;
				material.gpu_data.pbr_data.roughness = _asset.materials[ i ].pbrData.roughnessFactor;

				material.handle = _bindless_manager.add_buffer_entry( material.gpu_data );
			}
		}

		// TODO: remove unused meshes and materials
		const overloads make_transform(
			[]( const fastgltf::TRS& TRS )
			{
				const auto scale = glm::scale( glm::mat4( 1.f ), { TRS.scale.x(), TRS.scale.y(), TRS.scale.z() } );
				// glm quaternion constructor takes w first because why not?
				const auto rotation = glm::toMat4( glm::quat( TRS.rotation.w(), TRS.rotation.x(), TRS.rotation.y(), TRS.rotation.z() ) );
				const auto translation = glm::translate( glm::mat4( 1.f ),
														 { TRS.translation.x(), TRS.translation.y(), TRS.translation.z() } );
				return translation * rotation * scale;
			},
			[]( const fastgltf::math::fmat4x4& matrix )
			{
				return glm::mat4( matrix[ 0 ][ 0 ],
								  matrix[ 0 ][ 1 ],
								  matrix[ 0 ][ 2 ],
								  matrix[ 0 ][ 3 ],
								  matrix[ 1 ][ 0 ],
								  matrix[ 1 ][ 1 ],
								  matrix[ 1 ][ 2 ],
								  matrix[ 1 ][ 3 ],
								  matrix[ 2 ][ 0 ],
								  matrix[ 2 ][ 1 ],
								  matrix[ 2 ][ 2 ],
								  matrix[ 2 ][ 3 ],
								  matrix[ 3 ][ 0 ],
								  matrix[ 3 ][ 1 ],
								  matrix[ 3 ][ 2 ],
								  matrix[ 3 ][ 3 ] );
			} );

		std::vector<scene_node> nodes;
		nodes.reserve( _asset.nodes.size() );
		for ( const auto& node : _asset.nodes )
		{
			std::vector<uint32_t> children;
			children.reserve( node.children.size() );
			std::transform( begin( node.children ),
							end( node.children ),
							std::back_inserter( children ),
							[]( size_t i ) { return static_cast<uint32_t>( i ); } );
			nodes.push_back( scene_node { .children = std::move( children ),
										  .transform = std::visit( make_transform, node.transform ),
										  .mesh = static_cast<uint32_t>( node.meshIndex.value_or( -1 ) ) } );
		}

		std::vector<uint32_t> roots;
		roots.reserve( _asset.scenes[ 0 ].nodeIndices.size() );
		std::transform( begin( _asset.scenes[ 0 ].nodeIndices ),
						end( _asset.scenes[ 0 ].nodeIndices ),
						std::back_inserter( roots ),
						[]( size_t i ) { return static_cast<uint32_t>( i ); } );

		return asset { std::move( _meshes ),
					   std::move( _materials ),
					   glm::vec4( bounds.center[ 0 ], bounds.center[ 1 ], bounds.center[ 2 ], bounds.radius ),
					   std::move( nodes ),
					   std::move( roots ) };
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
	std::vector<uint8_t> _meshes_usage; // Could be a dynamic bitset
	std::vector<uint8_t> _images_usage; // Could be a dynamic bitset
	std::vector<texture_usage> _textures_usage;
	std::vector<uint8_t> _materials_usage; // Could be a dynamic bitset
	std::vector<mesh_data> _parsed_meshes;
	std::vector<mesh_upload_data> _mesh_upload_data;
	std::vector<data_source> _buffers;
	std::vector<sdl::raii::Surface> _images;
	std::vector<renderer::BindlessTexture> _textures;
	std::vector<material> _materials;
	std::vector<mesh> _meshes;
};

asset load_asset( renderer::Device& device, BindlessManager& bindless_manager, index_buffer& index_buffer, std::filesystem::path path )
{
	OPTICK_EVENT();
	asset_load_job job( device, bindless_manager, index_buffer, std::move( path ) );
	return job.run();
}
