#define GLM_ENABLE_EXPERIMENTAL

#include "asset.h"

#include <SDL3_image/SDL_image.h>
#include <fastgltf/core.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <istream>
#include <meshoptimizer.h>
#include <mikktspace.h>
#include <numeric>
#include <renderer/command_buffer.h>
#include <renderer/details/profiler.h>
#include <renderer/texture.h>
#include <renderer/third_party/tbb.h>

int calc_mip_levels( renderer::Extent2D extent )
{
	int mips = 1;

	while ( extent.width > 1 || extent.height > 1 )
	{
		++mips;
		extent.width /= 2;
		extent.height /= 2;
	}

	return mips;
}

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
	PROFILER_SCOPE();
	const auto format = sdl_to_vulkan_format( sdl_gpu_format( surface.format ), sRGB );
	if ( format == VK_FORMAT_UNDEFINED )
	{
		throw renderer::Error( "Unsupported format" );
	}
	using TexUsage = renderer::Texture::Usage;
	const renderer::Extent2D extent( surface.w, surface.h );
	return device.create_texture( renderer::Texture::Desc { .format = static_cast<renderer::Texture::Format>( format ),
															.usage = TexUsage::TRANSFER_SRC | TexUsage::TRANSFER_DST | TexUsage::SAMPLED,
															.extent = extent,
															.mips = calc_mip_levels( extent ) } );
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
		attribute_accessor( const std::byte* buffer, size_t count, size_t stride )
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

		const T& get( size_t index ) const { return reinterpret_cast<const T&>( _data[ index * _stride ] ); }
		size_t size() const { return _data.size(); }

	private:
		size_t _stride;
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
		return static_cast<int>( static_cast<UserData*>( context->m_pUserData )->indices.size() / 3 );
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
		, _upload_buffer( device.create_buffer( renderer::Buffer::Usage::TRANSFER_SRC, _size, renderer::Buffer::CpuUsage::WRITE ) )
		, _upload_fence( device.create_fence() )
	{
		_cmd_buffer->reset();
		_cmd_buffer->begin();
	}

	~gpu_upload_job() { assert( _state == state::DONE ); }

	void queue( const void* data, size_t size, renderer::Buffer& buffer, const size_t offset = 0 )
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
	void queue( std::span<const T> data, renderer::Buffer& buffer, const size_t offset = 0 )
	{
		assert( offset % alignof( T ) == 0 );
		queue( data.data(), data.size_bytes(), buffer, offset );
	}

	template <typename T>
	void queue( const std::vector<T>& data, renderer::Buffer& buffer, const size_t offset = 0 )
	{
		queue( std::span<const T>( data ), buffer, offset );
	}

	void queue( const SDL_Surface& surface, renderer::Texture& texture )
	{
		assert( _state == state::UPLOADING );
		const auto ptr = static_cast<std::byte*>( _upload_buffer.get_mapped_address() );
		const auto gpu_format = sdl_gpu_format( surface.format );
		const auto pitch = static_cast<int>( surface.w * get_format_size( sdl_to_vulkan_format( gpu_format ) ) );
		const auto size = static_cast<size_t>( pitch * surface.h );
		assert( size < _size );
		if ( size + _offset > _size )
		{
			flush_full();
		}
		if ( !SDL_ConvertPixels( surface.w, surface.h, surface.format, surface.pixels, surface.pitch, gpu_format, ptr + _offset, pitch ) )
		{
			throw renderer::Error( SDL_GetError() );
		}
		_cmd_buffer->texture_barrier( texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::TRANSFER_DST, 0 );
		_cmd_buffer->copy_buffer_to_texture( _upload_buffer, _offset, texture );
		for ( int mip = 1; mip < texture.get_mips(); ++mip )
		{
			_cmd_buffer->texture_barrier( texture, renderer::ResourceState::TRANSFER_DST, renderer::ResourceState::TRANSFER_SRC, mip - 1 );
			_cmd_buffer->texture_barrier( texture, renderer::ResourceState::UNDEFINED, renderer::ResourceState::TRANSFER_DST, mip );
			_cmd_buffer->blit_texture( texture, texture, mip - 1, mip, true );
			_cmd_buffer->texture_barrier( texture, renderer::ResourceState::TRANSFER_SRC, renderer::ResourceState::SHADER_READ, mip - 1 );
		}
		_cmd_buffer->texture_barrier( texture,
									  renderer::ResourceState::TRANSFER_DST,
									  renderer::ResourceState::SHADER_READ,
									  texture.get_mips() - 1 );
		_offset += size;
	}

	void submit()
	{
		assert( _state == state::UPLOADING );
		_cmd_buffer->memory_barrier( renderer::ResourceState::TRANSFER_DST, renderer::ResourceState::ANY_BUFFER_READ );
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

	size_t _size;
	renderer::Device& _device;
	renderer::raii::CommandBuffer _cmd_buffer;
	renderer::raii::Buffer _upload_buffer;
	renderer::raii::Fence _upload_fence;
	size_t _offset = 0;
	state _state = state::UPLOADING;
};

struct Assets::loaded_asset
{
	std::vector<mesh> meshes;
	std::vector<material> materials;
	glm::vec4 bounding_sphere;
	scene contents;
};

namespace
{
	glm::mat4 calc_node_transform( const fastgltf::Node& node )
	{
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
		return std::visit( make_transform, node.transform );
	}
}

class Assets::load_job
{
	struct mesh_attribute
	{
		std::size_t offset = 0;
		std::size_t stride = 0;
		std::size_t length = 0;
		VkFormat format = VK_FORMAT_UNDEFINED;
		uint32_t buffer_index = static_cast<uint32_t>( -1 );
	};

	struct mesh_data
	{
		mesh_attribute positions;
		mesh_attribute normals;
		mesh_attribute tangents;
		mesh_attribute uvs;
		mesh_attribute indices;
		size_t gltf_index;
	};

	// LODs simplified from a LOD 0 index buffer stored elsewhere
	struct lod_chain
	{
		std::vector<uint32_t> indices; // LOD 1+ indices, packed
		std::array<uint32_t, MESH_LOD_COUNT - 1> indices_per_lod;
		std::array<float, MESH_LOD_COUNT - 1> error_per_lod;
		std::array<uint32_t, MESH_LOD_COUNT> meshlets_per_lod; // Including LOD 0
	};

	struct mesh_upload_data
	{
		std::vector<mesh_vertex> vertices;
		std::span<const uint32_t> indices;
		lod_chain lods;
		// Depth-only rendering (shadows): vertices are only distinguished by position and LODs ignore attributes
		std::vector<uint32_t> shadow_indices;
		lod_chain shadow_lods;
		std::vector<uint32_t> expanded_indices;
		// Meshlets of all LODs, regular ones first then shadow ones
		std::vector<gpu_meshlet> meshlets;
		std::vector<uint32_t> meshlet_vertices;
		std::vector<uint8_t> meshlet_triangles;
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
	load_job( renderer::Device& device, BindlessManager& bindless_manager, index_buffer& index_buffer, std::filesystem::path path )
		: _device( device )
		, _bindless_manager( bindless_manager )
		, _index_buffer( index_buffer )
		, _path( std::move( path ) )
		, _mesh_upload( device, 32 * 1024 * 1024 )
		, _texture_upload( device, 128 * 1024 * 1024 )
	{
	}

	loaded_asset run()
	{
		PROFILER_SCOPE();
		parse();
		tbb::parallel_invoke(
			[ & ]()
			{
				prepare_meshes();
				upload_meshes();
			},
			[ & ] { decompress_textures(); } );
		// Can't upload texs while meshes are uploading, the command pool isn't thread safe...
		upload_textures();
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

	size_t get_scene_index() const { return _asset.defaultScene.value_or( 0 ); }

	void add_node( scene& contents, size_t node_index, const glm::mat4& parent_transform ) const
	{
		const auto& node = _asset.nodes[ node_index ];
		const auto transform = parent_transform * calc_node_transform( node );
		// Cameras and lights look down their node's -Z
		const auto forward = glm::normalize( glm::vec3( transform * glm::vec4( 0.f, 0.f, -1.f, 0.f ) ) );
		if ( node.cameraIndex && !contents.camera )
		{
			const auto perspective = std::get_if<fastgltf::Camera::Perspective>( &_asset.cameras[ *node.cameraIndex ].camera );
			if ( perspective )
			{
				contents.camera = Camera { .position = glm::vec3( transform[ 3 ] ),
										   .pitch = std::asin( glm::clamp( forward.y, -1.f, 1.f ) ),
										   .yaw = std::atan2( forward.x, -forward.z ),
										   .fov = perspective->yfov };
			}
		}
		if ( node.lightIndex && !contents.sun_direction && _asset.lights[ *node.lightIndex ].type == fastgltf::LightType::Directional )
		{
			contents.sun_direction = forward;
		}
		if ( node.meshIndex )
		{
			// glTF ignores a skinned mesh node's transform: its joints place it, in the bind pose at the scene root
			const auto mesh_transform = node.skinIndex ? glm::mat4( 1.f ) : transform;
			glm::vec3 scale, translation, skew;
			glm::vec4 perspective;
			glm::quat rotation;
			glm::decompose( mesh_transform, scale, rotation, translation, skew, perspective );
			contents.instances.push_back(
				{ .translation = translation, .rotation = rotation, .scale = scale, .mesh = _meshes[ *node.meshIndex ].handle } );
		}
		for ( const auto child : node.children )
		{
			add_node( contents, child, transform );
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
					default:
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
					default:
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
					default:
						break;
				}
				break;
			}
			default:
				break;
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
			overloads( []( const fastgltf::sources::Array& array )
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
		PROFILER_SCOPE();

		fastgltf::Parser parser( fastgltf::Extensions::KHR_materials_specular | fastgltf::Extensions::KHR_lights_punctual );
		fastgltf::GltfDataBuffer data;
		{
			PROFILER_SCOPE_N( "gltf_read_io" );
			auto res = fastgltf::GltfDataBuffer::FromPath( _path );
			if ( res.error() != fastgltf::Error::None )
			{
				return;
			}
			data = std::move( res.get() );
		}

		{
			PROFILER_SCOPE_N( "gltf_parse" );
			auto res = parser.loadGltf( data, _path.parent_path(), fastgltf::Options::None );
			if ( auto error = res.error(); error != fastgltf::Error::None )
			{
				return;
			}
			_asset = std::move( res.get() );
		}

		_meshes_usage.resize( _asset.meshes.size() );
		for ( const auto node : _asset.scenes[ get_scene_index() ].nodeIndices )
		{
			add_meshes( node );
		}

		_materials_usage.resize( _asset.materials.size(), 0 );

		for ( size_t i = 0; i < _meshes_usage.size(); ++i )
		{
			if ( _meshes_usage[ i ] )
			{
				auto& mesh = _asset.meshes[ i ];
				if ( mesh.primitives.size() != 1 )
				{
					throw renderer::Error( "Unsupported mesh with multiple primitives" );
				}
				if ( !mesh.primitives[ 0 ].materialIndex )
				{
					throw renderer::Error( "Unsupported mesh without material" );
				}
				_materials_usage[ *mesh.primitives[ 0 ].materialIndex ] = 1;
				auto& m = _parsed_meshes.emplace_back();
				m.gltf_index = i;

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
		for ( size_t i = 0; i < _materials_usage.size(); ++i )
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
		PROFILER_SCOPE();
		_mesh_upload_data.resize( _parsed_meshes.size() );
		tbb::parallel_for( 0zu, _parsed_meshes.size(), [ this ]( size_t index ) { prepare_mesh( static_cast<uint32_t>( index ) ); } );
#if 0
		for ( int i = 0; i < _parsed_meshes.size(); ++i )
		{
			prepare_mesh( i );
		}
#endif
	}

	void prepare_mesh( uint32_t mesh_index )
	{
		PROFILER_SCOPE();
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
			PROFILER_SCOPE_N( "generate_tangents" );
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
			PROFILER_SCOPE_N( "repack_vertices" );
			upload_data.vertices.reserve( parsed_mesh.positions.length );

			for ( size_t v = 0; v < parsed_mesh.positions.length; ++v )
			{
				const auto& uv = uvs.get( v );
				upload_data.vertices.emplace_back( positions.get( v ), uv.x, normals.get( v ), uv.y, tangents.get( v ) );
			}
		}

		{
			PROFILER_SCOPE_N( "lod_generation" );
			generate_lods( upload_data.vertices, upload_data.indices, upload_data.lods, true );
		}

		{
			PROFILER_SCOPE_N( "shadow_lod_generation" );
			// Simplify from the original indices: merging vertices by position first creates non-manifold edges (eg: duplicated
			// back faces) which stop the simplifier early. Attribute seams don't matter for depth-only rendering though, so every
			// LOD is remapped afterwards to only distinguish vertices by position (fewer unique vertices to transform)
			generate_lods( upload_data.vertices, upload_data.indices, upload_data.shadow_lods, false );
			upload_data.shadow_indices.assign( begin( upload_data.indices ), end( upload_data.indices ) );
			remap_to_shadow_indices( upload_data.vertices, upload_data.shadow_indices );
			size_t lod_offset = 0;
			for ( const auto lod_index_count : upload_data.shadow_lods.indices_per_lod )
			{
				remap_to_shadow_indices( upload_data.vertices,
										 std::span( upload_data.shadow_lods.indices ).subspan( lod_offset, lod_index_count ) );
				lod_offset += lod_index_count;
			}
		}

		{
			PROFILER_SCOPE_N( "meshlet_generation" );
			build_meshlets( upload_data, upload_data.indices, upload_data.lods );
			build_meshlets( upload_data, upload_data.shadow_indices, upload_data.shadow_lods );
		}
	}

	// Simplify lod0_indices into the LODs 1+ of the chain
	// Attributes (normals) are only taken into account if use_attributes is set, which is meant for regular (shaded) rendering
	static void
	generate_lods( std::span<const mesh_vertex> vertices, std::span<const uint32_t> lod0_indices, lod_chain& chain, bool use_attributes )
	{
		// Each LOD is as coarse as its error limit allows (no index count target), limits double from one LOD to the next
		// so that the LOD selected for a pixel tolerance never exceeds twice the error it could afford
		constexpr float first_lod_error = 0.0025f;
		std::vector<uint32_t> buffer( lod0_indices.size() );
		// Simplification errors (targets and results) are relative to the mesh extents, but LOD selection compares them
		// against world space distances (in mesh units, before instance scale) so convert results to absolute
		const float error_scale = meshopt_simplifyScale( &vertices.front().position.x, vertices.size(), sizeof( mesh_vertex ) );
		for ( size_t lod = 0; lod < chain.indices_per_lod.size(); ++lod )
		{
			const float target_error = first_lod_error * static_cast<float>( 1 << lod );
			size_t size = 0;
			if ( use_attributes )
			{
				const float attr_weights[ 3 ] = { 0.5f, 0.5f, 0.5f };
				size = meshopt_simplifyWithAttributes( buffer.data(),
													   lod0_indices.data(),
													   lod0_indices.size(),
													   &vertices.front().position.x,
													   vertices.size(),
													   sizeof( mesh_vertex ),
													   &vertices.front().normal.x,
													   sizeof( mesh_vertex ),
													   attr_weights,
													   3,
													   nullptr,
													   0,
													   target_error,
													   0,
													   &chain.error_per_lod[ lod ] );
			}
			else
			{
				size = meshopt_simplify( buffer.data(),
										 lod0_indices.data(),
										 lod0_indices.size(),
										 &vertices.front().position.x,
										 vertices.size(),
										 sizeof( mesh_vertex ),
										 0,
										 target_error,
										 0,
										 &chain.error_per_lod[ lod ] );
			}
			chain.error_per_lod[ lod ] *= error_scale;

			meshopt_optimizeVertexCache( buffer.data(), buffer.data(), size, vertices.size() );
			chain.indices_per_lod[ lod ] = static_cast<uint32_t>( size );
			chain.indices.insert( end( chain.indices ), buffer.data(), buffer.data() + size );
		}
	}

	// Remap indices in place so vertices are only distinguished by their position, then optimize for vertex cache again
	static void remap_to_shadow_indices( std::span<const mesh_vertex> vertices, std::span<uint32_t> indices )
	{
		if ( indices.empty() )
		{
			return;
		}
		meshopt_generateShadowIndexBuffer( indices.data(),
										   indices.data(),
										   indices.size(),
										   &vertices.front().position.x,
										   vertices.size(),
										   sizeof( glm::vec3 ),
										   sizeof( mesh_vertex ) );
		meshopt_optimizeVertexCache( indices.data(), indices.data(), indices.size(), vertices.size() );
	}

	// Build meshlets for every LOD of the chain (starting with lod0_indices) and append them to the mesh meshlets
	static void build_meshlets( mesh_upload_data& upload_data, std::span<const uint32_t> lod0_indices, lod_chain& chain )
	{
		std::vector<meshopt_Meshlet> meshlet_buffer;
		size_t index_offset = 0;
		for ( int lod = 0; lod < MESH_LOD_COUNT; ++lod )
		{
			const auto indices_ptr = lod == 0 ? lod0_indices.data() : chain.indices.data() + index_offset;
			const auto indices_count = lod == 0 ? lod0_indices.size() : chain.indices_per_lod[ lod - 1 ];
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

				chain.meshlets_per_lod[ lod ] = static_cast<uint32_t>( meshlet_count );
				const auto& last = meshlet_buffer[ meshlet_count - 1 ];

				upload_data.meshlet_vertices.resize( vertices_offset + last.vertex_offset + last.vertex_count );
				upload_data.meshlet_triangles.resize( triangles_offset + last.triangle_offset + last.triangle_count * 3 );

				for ( const auto& meshlet : std::span( meshlet_buffer.data(), meshlet_count ) )
				{
					const size_t vertex_index = meshlet.vertex_offset + vertices_offset;
					const size_t triangle_index = meshlet.triangle_offset + triangles_offset;
					const auto bounds = meshopt_computeMeshletBounds( upload_data.meshlet_vertices.data() + vertex_index,
																	  upload_data.meshlet_triangles.data() + triangle_index,
																	  meshlet.triangle_count,
																	  &upload_data.vertices.front().position.x,
																	  upload_data.vertices.size(),
																	  sizeof( mesh_vertex ) );

					upload_data.meshlets.emplace_back(
						static_cast<uint32_t>( vertex_index ),
						static_cast<uint32_t>( triangle_index ),
						meshlet.vertex_count,
						meshlet.triangle_count,
						glm::vec4( bounds.center[ 0 ], bounds.center[ 1 ], bounds.center[ 2 ], bounds.radius ),
						glm::vec4( bounds.cone_axis[ 0 ], bounds.cone_axis[ 1 ], bounds.cone_axis[ 2 ], bounds.cone_cutoff ) );
				}
			}
			else
			{
				chain.meshlets_per_lod[ lod ] = 0;
			}
		}
	}

	void upload_meshes()
	{
		PROFILER_SCOPE();
		_meshes.resize( _asset.meshes.size() );
		for ( size_t i = 0; i < _parsed_meshes.size(); ++i )
		{
			upload_mesh( i );
		}
		_mesh_upload.submit();
	}

	void upload_mesh( size_t mesh_index )
	{
		PROFILER_SCOPE();
		const auto& upload_data = _mesh_upload_data[ mesh_index ];
		auto& mesh = _meshes[ _parsed_meshes[ mesh_index ].gltf_index ];

		const auto bounds = meshopt_computeSphereBounds( &upload_data.vertices[ 0 ].position.x,
														 upload_data.vertices.size(),
														 sizeof( mesh_vertex ),
														 nullptr,
														 0 );
		mesh.gpu_data.bounding_sphere.x = bounds.center[ 0 ];
		mesh.gpu_data.bounding_sphere.y = bounds.center[ 1 ];
		mesh.gpu_data.bounding_sphere.z = bounds.center[ 2 ];
		mesh.gpu_data.bounding_sphere.w = bounds.radius;

		// Index buffer + LOD descriptors, shadow meshlets are stored after the regular ones
		mesh.meshlets_count = upload_lods( upload_data.indices, upload_data.lods, 0, mesh.gpu_data.lods );
		upload_lods( upload_data.shadow_indices, upload_data.shadow_lods, mesh.meshlets_count, mesh.gpu_data.shadow_lods );
		mesh.max_task_groups = 0;
		for ( const auto chain : { &upload_data.lods, &upload_data.shadow_lods } )
		{
			for ( const auto meshlets : chain->meshlets_per_lod )
			{
				mesh.max_task_groups = std::max( mesh.max_task_groups, ( meshlets + TASK_GROUP_SIZE - 1 ) / TASK_GROUP_SIZE );
			}
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

		mesh.gpu_data.meshlet_vertices = mesh.data_buffer.get_device_address() + offset;
		_mesh_upload.queue( upload_data.meshlet_vertices, mesh.data_buffer, offset );
		offset += upload_data.meshlet_vertices.size() * sizeof( uint32_t );

		mesh.gpu_data.meshlet_triangles = mesh.data_buffer.get_device_address() + offset;
		_mesh_upload.queue( upload_data.meshlet_triangles, mesh.data_buffer, offset );
		offset += upload_data.meshlet_triangles.size() * sizeof( uint8_t );
	}

	// Upload the chain indices (LOD 0 first) to the index buffer and fill the matching LOD descriptors
	// Meshlets of the chain are expected to be contiguous, starting at first_meshlet
	// Returns the meshlet offset following the chain
	uint32_t upload_lods( std::span<const uint32_t> lod0_indices,
						  const lod_chain& chain,
						  uint32_t first_meshlet,
						  std::array<gpu_mesh_lod, MESH_LOD_COUNT>& lods )
	{
		const size_t index_count = lod0_indices.size() + chain.indices.size();
		auto index_offset = static_cast<uint32_t>( _index_buffer.size.fetch_add( index_count ) );
		if ( index_offset + index_count > _index_buffer.capacity )
		{
			throw renderer::Error( "Index buffer is full" );
		}
		_mesh_upload.queue( lod0_indices, _index_buffer.buffer, index_offset * sizeof( uint32_t ) );
		if ( !chain.indices.empty() )
		{
			_mesh_upload.queue( chain.indices, _index_buffer.buffer, ( index_offset + lod0_indices.size() ) * sizeof( uint32_t ) );
		}

		auto meshlet_offset = first_meshlet;
		for ( size_t lod = 0; lod < MESH_LOD_COUNT; ++lod )
		{
			lods[ lod ] = gpu_mesh_lod { .indices_offset = index_offset,
										 .indices_count = lod == 0 ? static_cast<uint32_t>( lod0_indices.size() )
																   : chain.indices_per_lod[ lod - 1 ],
										 .meshlets_offset = meshlet_offset,
										 .meshlets_count = chain.meshlets_per_lod[ lod ],
										 .error = lod == 0 ? 0.f : chain.error_per_lod[ lod - 1 ] };
			index_offset += lods[ lod ].indices_count;
			meshlet_offset += lods[ lod ].meshlets_count;
		}
		return meshlet_offset;
	}

	void decompress_textures()
	{
		PROFILER_SCOPE();
		_images.resize( _images_usage.size() );

		tbb::parallel_for(
			0uz,
			_images_usage.size(),
			[ & ]( size_t index )
			{
				PROFILER_SCOPE_N( "decompress_tex" );
				if ( _images_usage[ index ] )
				{
					const auto source = load( _asset.images[ index ].data, true );
					_images[ index ].reset( IMG_Load_IO( SDL_IOFromConstMem( source.data.data(), source.data.size() ), true ) );
					if ( !_images[ index ] )
					{
						throw renderer::Error( SDL_GetError() );
					}
					// XXX: INDEX1MSB should probably be turned into R8_UNORM
					else if ( _images[ index ]->format == SDL_PIXELFORMAT_INDEX8 || _images[ index ]->format == SDL_PIXELFORMAT_INDEX1MSB )
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
		PROFILER_SCOPE();
		_textures.resize( _textures_usage.size() );
		for ( size_t i = 0; i < _textures_usage.size(); ++i )
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

	loaded_asset finalize()
	{
		PROFILER_SCOPE();
		std::vector<glm::vec4> bounding_spheres;
		bounding_spheres.reserve( _parsed_meshes.size() );
		for ( const auto& parsed_mesh : _parsed_meshes )
		{
			bounding_spheres.push_back( _meshes[ parsed_mesh.gltf_index ].gpu_data.bounding_sphere );
		}
		const auto bounds = meshopt_computeSphereBounds( &bounding_spheres[ 0 ].x,
														 bounding_spheres.size(),
														 sizeof( glm::vec4 ),
														 &bounding_spheres[ 0 ].w,
														 sizeof( glm::vec4 ) );

		gpu_upload_job::wait_all_done( { &_mesh_upload, &_texture_upload } );

		_materials.resize( _asset.materials.size() );
		for ( size_t i = 0; i < _materials_usage.size(); ++i )
		{
			if ( _materials_usage[ i ] )
			{
				auto& material = _materials[ i ];
				if ( _asset.materials[ i ].pbrData.baseColorTexture )
				{
					material.gpu_data.albedo_texture = _textures[ _asset.materials[ i ].pbrData.baseColorTexture->textureIndex ]
														   .handles.sampled_texture;
				}

				if ( _asset.materials[ i ].normalTexture )
				{
					material.gpu_data.normal_texture = _textures[ _asset.materials[ i ].normalTexture->textureIndex ]
														   .handles.sampled_texture;
				}

				if ( _asset.materials[ i ].pbrData.metallicRoughnessTexture )
				{
					material.gpu_data
						.metallic_roughness_texture = _textures[ _asset.materials[ i ].pbrData.metallicRoughnessTexture->textureIndex ]
														  .handles.sampled_texture;
				}
				if ( _asset.materials[ i ].specular )
				{
					if ( _asset.materials[ i ].specular->specularTexture )
					{
						material.gpu_data.specular_texture = _textures[ _asset.materials[ i ].specular->specularTexture->textureIndex ]
																 .handles.sampled_texture;
					}
				}
				material.gpu_data.base_color.x = _asset.materials[ i ].pbrData.baseColorFactor.x();
				material.gpu_data.base_color.y = _asset.materials[ i ].pbrData.baseColorFactor.y();
				material.gpu_data.base_color.z = _asset.materials[ i ].pbrData.baseColorFactor.z();
				material.gpu_data.base_color.w = _asset.materials[ i ].pbrData.baseColorFactor.w();
				material.gpu_data.metalness = _asset.materials[ i ].pbrData.metallicFactor;
				material.gpu_data.roughness = _asset.materials[ i ].pbrData.roughnessFactor;
				switch ( _asset.materials[ i ].alphaMode )
				{

					case fastgltf::AlphaMode::Opaque:
					case fastgltf::AlphaMode::Mask: // TODO: proper Mask support
						material.gpu_data.type = _asset.materials[ i ].doubleSided ? material_type::OPAQUE_DOUBLE
																				   : material_type::OPAQUE_SINGLE;
						break;
					case fastgltf::AlphaMode::Blend:
						material.gpu_data.type = _asset.materials[ i ].doubleSided ? material_type::ALPHA_DOUBLE
																				   : material_type::ALPHA_SINGLE;
						break;
					default:
						throw renderer::Error( "Unsupported material alpha mode" );
				}
				material.type = material.gpu_data.type;

				material.handle = _bindless_manager.add_buffer_entry( material.gpu_data );
			}
		}

		for ( const auto& parsed_mesh : _parsed_meshes )
		{
			auto& mesh = _meshes[ parsed_mesh.gltf_index ];
			mesh.handle = _bindless_manager.add_buffer_entry( mesh.gpu_data );
			mesh.material = _materials[ *_asset.meshes[ parsed_mesh.gltf_index ].primitives[ 0 ].materialIndex ].handle;
		}

		scene contents;
		for ( const auto root : _asset.scenes[ get_scene_index() ].nodeIndices )
		{
			add_node( contents, root, glm::mat4( 1.f ) );
		}

		// Both arrays match the glTF ones, with gaps for what isn't used: keep only what was loaded, in handle order
		std::vector<mesh> meshes;
		meshes.reserve( _parsed_meshes.size() );
		for ( auto& mesh : _meshes )
		{
			if ( mesh.handle )
			{
				meshes.push_back( std::move( mesh ) );
			}
		}
		std::vector<material> materials;
		for ( auto& material : _materials )
		{
			if ( material.handle )
			{
				materials.push_back( std::move( material ) );
			}
		}

		return loaded_asset { std::move( meshes ),
							  std::move( materials ),
							  glm::vec4( bounds.center[ 0 ], bounds.center[ 1 ], bounds.center[ 2 ], bounds.radius ),
							  std::move( contents ) };
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

Assets::Assets( renderer::Device& device, BindlessManager& bindless_manager, size_t max_indices )
	: _device( device )
	, _bindless_manager( bindless_manager )
	, _indices( device, max_indices )
{
}

scene Assets::load( const std::filesystem::path& path )
{
	PROFILER_SCOPE();
	load_job job( _device, _bindless_manager, _indices, path );
	auto asset = job.run();
	add_content( asset );
	return std::move( asset.contents );
}

void Assets::add_content( loaded_asset& asset )
{
	for ( auto& material : asset.materials )
	{
		// This assert should hold unless materials are loaded through another path
		// If that becomes the case we should revisit our architecture
		assert( material.handle.index == _materials.size() );
		_materials.emplace_back( std::move( material ) );
	}
	for ( auto& mesh : asset.meshes )
	{
		// Same as for materials
		assert( mesh.handle.index == _meshes.size() );
		_meshes.emplace_back( std::move( mesh ) );
	}
}
