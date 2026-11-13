#define GLM_ENABLE_EXPERIMENTAL

#include "asset.h"

#include "dds_parser.h"
#include "texture_compressor.h"

#include <SDL3_image/SDL_image.h>
#include <algorithm>
#include <bit>
#include <exception>
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

constexpr glm::vec3 make_vec3( const fastgltf::math::nvec3& vec )
{
	return { vec.x(), vec.y(), vec.z() };
}

constexpr glm::vec4 make_vec4( const fastgltf::math::nvec4& vec )
{
	return { vec.x(), vec.y(), vec.z(), vec.w() };
}

constexpr glm::vec4 make_vec4( const fastgltf::math::nvec3& vec, float w )
{
	return { vec.x(), vec.y(), vec.z(), w };
}

struct mesh_vertex
{
	glm::vec3 position;
	float u;
	glm::vec3 normal;
	float v;
	glm::vec4 tangent;
};

renderer::raii::Texture make_image_from_surface( renderer::Device& device, const SDL_Surface& surface, renderer::Texture::Format format )
{
	PROFILER_SCOPE();
	using TexUsage = renderer::Texture::Usage;
	const renderer::Extent2D extent( surface.w, surface.h );
	return device.create_texture( renderer::Texture::Desc { .format = format,
															.usage = TexUsage::TRANSFER_SRC | TexUsage::TRANSFER_DST | TexUsage::SAMPLED,
															.extent = extent,
															.mips = calc_mip_levels( extent ) } );
}

renderer::raii::Texture make_image_from_dds( renderer::Device& device, const DDSFile& dds )
{
	PROFILER_SCOPE();
	auto desc = dds.get_desc();
	desc.usage = renderer::Texture::Usage::TRANSFER_DST | renderer::Texture::Usage::SAMPLED;
	return device.create_texture( desc );
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
	// compressed_size is the room for texture blocks compressed on the GPU, 0 if the job doesn't compress any
	gpu_upload_job( renderer::Device& device, uint32_t max_size, uint32_t compressed_size = 0 )
		: _size( max_size )
		, _compressed_size( compressed_size )
		, _device( device )
		, _cmd_buffer( device.grab_command_buffer() )
		, _upload_buffer( device.create_buffer( renderer::Buffer::Usage::TRANSFER_SRC, _size, renderer::Buffer::CpuUsage::WRITE ) )
		, _upload_fence( device.create_fence() )
	{
		_cmd_buffer->reset();
		_cmd_buffer->begin();
		if ( _compressed_size != 0 )
		{
			using BufUsage = renderer::Buffer::Usage;
			_compressed_buffer = device.create_buffer( BufUsage::STORAGE_BUFFER | BufUsage::TRANSFER_SRC | BufUsage::SHADER_DEVICE_ADDRESS,
													   _compressed_size );
		}
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

	// The surface must be RGBA32
	void queue( const SDL_Surface& surface, renderer::Texture& texture )
	{
		assert( _state == state::UPLOADING );
		const auto size = surface_size( surface );
		assert( size < _size );
		align_offset( renderer::Texture::get_texel_block_size( texture.get_format() ) );
		if ( size + _offset > _size )
		{
			flush_full();
		}
		copy_surface( surface );
		_cmd_buffer->transition_texture( texture, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL );
		_cmd_buffer->copy_buffer_to_texture( _upload_buffer, _offset, texture );
		for ( int mip = 1; mip < texture.get_mips(); ++mip )
		{
			_cmd_buffer->barrier( renderer::Stage::TRANSFER, renderer::Stage::TRANSFER );
			_cmd_buffer->blit_texture( texture, texture, mip - 1, mip, true );
		}
		_offset += size;
	}

	// The surface must be RGBA32 and meet the compressor's requirements. Texture is the final block compressed texture.
	void queue( const SDL_Surface& surface, renderer::Texture& texture, const texture_compressor& compressor )
	{
		assert( _state == state::UPLOADING );
		const auto size = surface_size( surface );
		assert( size < _size );
		size_t compressed_size = 0;
		for ( int mip = 0; mip < texture.get_mips(); ++mip )
		{
			compressed_size += texture.get_mip_size( mip );
		}
		assert( compressed_size <= _compressed_size );
		const auto block_size = renderer::Texture::get_texel_block_size( texture.get_format() );
		align_offset( renderer::Texture::get_texel_block_size( renderer::Texture::Format::R8G8B8A8_UNORM ) );
		_compressed_offset = ( _compressed_offset + block_size - 1 ) / block_size * block_size;
		if ( size + _offset > _size || compressed_size + _compressed_offset > _compressed_size )
		{
			flush_full();
		}
		// Recording first: compress() rejects a texture too large before it's copied, the copy is only read at submit
		compressor.compress( *_cmd_buffer, _upload_buffer, _offset, _compressed_buffer, _compressed_offset, texture );
		copy_surface( surface );
		_offset += size;
		_compressed_offset += compressed_size;
	}

	// DDS files come with their mips, copied as is
	void queue( const DDSFile& dds, renderer::Texture& texture )
	{
		assert( _state == state::UPLOADING );
		const auto& desc = dds.get_desc();
		const auto size = desc.get_size();
		assert( size < _size );
		// Mip sizes are multiples of the block size, so aligning the first one aligns them all
		align_offset( renderer::Texture::get_texel_block_size( texture.get_format() ) );
		if ( size + _offset > _size )
		{
			flush_full();
		}
		const auto ptr = static_cast<std::byte*>( _upload_buffer.get_mapped_address() );
		memcpy( ptr + _offset, dds.get_data().data(), size );
		_cmd_buffer->transition_texture( texture, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::GENERAL );
		for ( int mip = 0; mip < texture.get_mips(); ++mip )
		{
			_cmd_buffer->copy_buffer_to_texture( _upload_buffer, _offset, texture, mip );
			_offset += desc.get_mip_size( mip );
		}
	}

	// Submits what was queued so far and waits for it, if anything was. The job keeps uploading afterwards.
	void flush()
	{
		if ( _offset != 0 || _compressed_offset != 0 )
		{
			flush_full();
		}
	}

	void submit()
	{
		assert( _state == state::UPLOADING );
		_cmd_buffer->barrier( renderer::Stage::TRANSFER, renderer::Stage::INDIRECT | renderer::Stage::INDEX | renderer::Stage::ANY_SHADER );
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
	void align_offset( size_t alignment ) { _offset = ( _offset + alignment - 1 ) / alignment * alignment; }

	static constexpr size_t RGBA32_SIZE = 4;
	static size_t surface_size( const SDL_Surface& surface )
	{
		return static_cast<size_t>( surface.w ) * static_cast<size_t>( surface.h ) * RGBA32_SIZE;
	}

	// Copies the surface at the current offset
	void copy_surface( const SDL_Surface& surface )
	{
		PROFILER_SCOPE();
		assert( surface.format == SDL_PIXELFORMAT_RGBA32 );
		assert( static_cast<size_t>( surface.pitch ) == surface.w * RGBA32_SIZE );
		memcpy( static_cast<std::byte*>( _upload_buffer.get_mapped_address() ) + _offset, surface.pixels, surface_size( surface ) );
	}

	void flush_full()
	{
		submit();
		// XXX: Could use 2 alternating buffers instead of one to avoid sync waiting
		wait_done();
		_device.reset_fences( { _upload_fence } );
		_offset = 0;
		_compressed_offset = 0;
		_state = state::UPLOADING;
		_cmd_buffer->reset();
		_cmd_buffer->begin();
	}

	size_t _size;
	size_t _compressed_size;
	renderer::Device& _device;
	renderer::raii::CommandBuffer _cmd_buffer;
	renderer::raii::Buffer _upload_buffer;
	renderer::raii::Buffer _compressed_buffer;
	renderer::raii::Fence _upload_fence;
	size_t _offset = 0;
	size_t _compressed_offset = 0;
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
	// What uploading needs, owned by Assets::upload_prepared() and shared by every job
	struct upload_state
	{
		renderer::Device& device;
		BindlessManager& bindless_manager;
		index_buffer& indices;
		texture_compressor* compressor; // Null without compression
		gpu_upload_job mesh_upload;
		gpu_upload_job texture_upload;
	};
}

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

	// Meshlets of some LODs of one chain, built separately then merged into mesh_upload_data
	struct meshlet_set
	{
		std::vector<gpu_meshlet> meshlets; // Offsets relative to vertices and triangles below
		std::vector<uint32_t> vertices;
		std::vector<uint8_t> triangles;
	};

	// How the shaders read a texture, which decides its formats
	enum class texture_usage
	{
		UNUSED,
		ALBEDO,
		ALBEDO_ALPHA, // Alpha tested or blended
		NORMAL,
		METALLIC_ROUGHNESS,
		OCCLUSION,
		OCCLUSION_METALLIC_ROUGHNESS, // One texture referenced with both usages
		EMISSIVE,
		SPECULAR_GLOSSINESS,
		SPECULAR // KHR_materials_specular's factor, in alpha
	};

	static bool is_srgb( texture_usage usage )
	{
		return usage == texture_usage::ALBEDO || usage == texture_usage::ALBEDO_ALPHA || usage == texture_usage::EMISSIVE
			|| usage == texture_usage::SPECULAR_GLOSSINESS;
	}

	static renderer::Texture::Format uncompressed_format( texture_usage usage )
	{
		return is_srgb( usage ) ? renderer::Texture::Format::R8G8B8A8_SRGB : renderer::Texture::Format::R8G8B8A8_UNORM;
	}

	static renderer::Texture::Format compressed_format( texture_usage usage )
	{
		using Format = renderer::Texture::Format;
		switch ( usage )
		{
			case texture_usage::ALBEDO:
			case texture_usage::EMISSIVE:
				return Format::BC1_RGB_SRGB;
			case texture_usage::ALBEDO_ALPHA:
			case texture_usage::SPECULAR_GLOSSINESS:
				return Format::BC3_SRGB;
			case texture_usage::NORMAL:
				return Format::BC5_UNORM;
			case texture_usage::METALLIC_ROUGHNESS:
			case texture_usage::OCCLUSION:
			case texture_usage::OCCLUSION_METALLIC_ROUGHNESS:
				return Format::BC1_RGB_UNORM;
			case texture_usage::SPECULAR:
				return Format::BC3_UNORM;
			case texture_usage::UNUSED:
				break;
		}
		throw renderer::Error( "Unused texture has no format" );
	}

	// The usage of a texture referenced with both, if it can serve them
	static std::optional<texture_usage> merge_usages( texture_usage a, texture_usage b )
	{
		const auto is_any_of = []( texture_usage usage, std::initializer_list<texture_usage> usages )
		{ return std::find( begin( usages ), end( usages ), usage ) != end( usages ); };
		if ( a == b )
		{
			return a;
		}
		// These all read RGB as sRGB, so one image can serve them, keeping alpha if one of them needs it. eg: an opaque and a
		// blended material sharing a texture, or a glowing sign reusing its albedo and cutout alpha as emissive
		const std::initializer_list<texture_usage> color { texture_usage::ALBEDO, texture_usage::ALBEDO_ALPHA, texture_usage::EMISSIVE };
		if ( is_any_of( a, color ) && is_any_of( b, color ) )
		{
			return a == texture_usage::ALBEDO_ALPHA || b == texture_usage::ALBEDO_ALPHA ? texture_usage::ALBEDO_ALPHA
																						: texture_usage::ALBEDO;
		}
		const std::initializer_list<texture_usage> orm { texture_usage::METALLIC_ROUGHNESS,
														 texture_usage::OCCLUSION,
														 texture_usage::OCCLUSION_METALLIC_ROUGHNESS };
		if ( is_any_of( a, orm ) && is_any_of( b, orm ) )
		{
			return texture_usage::OCCLUSION_METALLIC_ROUGHNESS;
		}
		return std::nullopt;
	}

	static texture_usage albedo_usage( const fastgltf::Material& material )
	{
		return material.alphaMode == fastgltf::AlphaMode::Opaque ? texture_usage::ALBEDO : texture_usage::ALBEDO_ALPHA;
	}

	enum class image_usage : uint8_t
	{
		UNUSED,
		DECODED, // Decoded by SDL_image, mips generated on the GPU
		DDS
	};

	struct data_source
	{
		std::span<const std::byte> data;
		std::unique_ptr<std::byte[]> buffer;
	};

	// A decoded surface, or a DDS file uploaded as is and parsed once its sRGB usage is known
	using loaded_image = std::variant<std::monostate, sdl::raii::Surface, data_source>;

	using gpu_state = upload_state;

public:
	// Prepared meshes and textures are pushed to uploads
	load_job( bool use_dds, std::filesystem::path path, upload_queue& uploads )
		: _use_dds( use_dds )
		, _path( std::move( path ) )
		, _uploads( uploads )
	{
	}

	// Reads, parses and processes the file, doesn't touch the device. Errors are kept for get_error() instead of thrown:
	// a throwing task would cancel the task group and skip the other jobs' tasks, which would never push JOB_DONE.
	void prepare()
	{
		PROFILER_SCOPE();
		try
		{
			parse();
			run_prepare_graph();
		}
		catch ( ... )
		{
			_error = std::current_exception();
		}
		// The graph is done: every item of this job was pushed before it
		_uploads.push( { upload_item::kind::JOB_DONE, this } );
	}

	const std::exception_ptr& get_error() const { return _error; }

	// Uploads an item this job pushed, then frees its CPU side data. From Assets::upload_prepared() only.
	void upload( gpu_state& gpu, const upload_item& item )
	{
		if ( item.kind == upload_item::kind::MESH )
		{
			upload_mesh( gpu, item.index );
			_mesh_upload_data[ item.index ] = { };
		}
		else
		{
			upload_texture( gpu, item.index );
			// No other texture uses this image (add_texture_usage() rejects that), so it can be freed now
			_images[ get_image_index( _asset.textures[ item.index ] ) ] = std::monostate { };
		}
	}

	// Adds the materials and meshes to the bindless manager and builds the scene. Once every item of the job is
	// uploaded, and in load order: handles must follow the loaded materials and meshes.
	loaded_asset finalize( BindlessManager& bindless_manager ) { return finalize_impl( bindless_manager ); }

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

	// Prefers the DDS image when there is one, block compressed with precomputed mips, unless it isn't wanted and there's
	// another image
	size_t get_image_index( const fastgltf::Texture& texture ) const
	{
		return texture.ddsImageIndex && ( _use_dds || !texture.imageIndex ) ? *texture.ddsImageIndex : *texture.imageIndex;
	}

	void add_texture_usage( size_t index, texture_usage usage )
	{
		if ( _textures_usage[ index ] == texture_usage::UNUSED )
		{
			const auto& texture = _asset.textures[ index ];
			const auto image_index = get_image_index( texture );
			if ( _images_usage[ image_index ] != image_usage::UNUSED )
			{
				throw renderer::Error( "Unsupported multiple texture references to the same image" );
			}
			_images_usage[ image_index ] = texture.ddsImageIndex && *texture.ddsImageIndex == image_index ? image_usage::DDS
																										  : image_usage::DECODED;
			_textures_usage[ index ] = usage;
		}
		else if ( const auto merged = merge_usages( _textures_usage[ index ], usage ) )
		{
			_textures_usage[ index ] = *merged;
		}
		else
		{
			throw renderer::Error( "Unsupported use of a texture with incompatible usages" );
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
							   PROFILER_SCOPE_N( "read_asset" );
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

		fastgltf::Parser parser( fastgltf::Extensions::KHR_lights_punctual | fastgltf::Extensions::KHR_materials_emissive_strength
								 | fastgltf::Extensions::KHR_materials_pbrSpecularGlossiness | fastgltf::Extensions::KHR_materials_specular
								 | fastgltf::Extensions::MSFT_texture_dds );
		fastgltf::GltfDataBuffer data;
		{
			PROFILER_SCOPE_N( "gltf_read_io" );
			auto res = fastgltf::GltfDataBuffer::FromPath( _path );
			if ( res.error() != fastgltf::Error::None )
			{
				throw renderer::Error( std::format( "GLTF parser error: {}", fastgltf::getErrorMessage( res.error() ) ) );
			}
			data = std::move( res.get() );
		}

		{
			PROFILER_SCOPE_N( "gltf_parse" );
			auto res = parser.loadGltf( data, _path.parent_path(), fastgltf::Options::None );
			if ( auto error = res.error(); error != fastgltf::Error::None )
			{
				throw renderer::Error( std::format( "GLTF parser error: {}", fastgltf::getErrorMessage( res.error() ) ) );
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

		_images_usage.resize( _asset.images.size(), image_usage::UNUSED );
		_textures_usage.resize( _asset.textures.size(), texture_usage::UNUSED );
		for ( size_t i = 0; i < _materials_usage.size(); ++i )
		{
			if ( _materials_usage[ i ] )
			{
				const auto& material = _asset.materials[ i ];
				if ( material.specularGlossiness )
				{
					// Both workflows fill base_color and albedo_texture: only one can be used. glTF allows a metallic-roughness
					// fallback next to specular-glossiness, but we can't tell its factors from the defaults
					if ( material.pbrData.baseColorTexture || material.pbrData.metallicRoughnessTexture || material.specular )
					{
						throw renderer::Error( "Unsupported material with both metallic-roughness and specular-glossiness" );
					}
					if ( material.specularGlossiness->diffuseTexture )
					{
						add_texture_usage( material.specularGlossiness->diffuseTexture->textureIndex, albedo_usage( material ) );
					}
					if ( material.specularGlossiness->specularGlossinessTexture )
					{
						add_texture_usage( material.specularGlossiness->specularGlossinessTexture->textureIndex,
										   texture_usage::SPECULAR_GLOSSINESS );
					}
				}
				else
				{
					if ( material.pbrData.baseColorTexture )
					{
						add_texture_usage( material.pbrData.baseColorTexture->textureIndex, albedo_usage( material ) );
					}
					if ( material.pbrData.metallicRoughnessTexture )
					{
						add_texture_usage( material.pbrData.metallicRoughnessTexture->textureIndex, texture_usage::METALLIC_ROUGHNESS );
					}
					if ( material.specular )
					{
						if ( material.specular->specularTexture )
						{
							add_texture_usage( material.specular->specularTexture->textureIndex, texture_usage::SPECULAR );
						}
					}
				}
				if ( material.normalTexture )
				{
					add_texture_usage( material.normalTexture->textureIndex, texture_usage::NORMAL );
				}
				if ( material.occlusionTexture )
				{
					add_texture_usage( material.occlusionTexture->textureIndex, texture_usage::OCCLUSION );
				}
				if ( material.emissiveTexture )
				{
					add_texture_usage( material.emissiveTexture->textureIndex, texture_usage::EMISSIVE );
				}
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

	// Meshes and textures as one flow graph: a step runs once its predecessors are done and never waits for anything, so a
	// thread can't stack an unrelated mesh on top of one it's waiting for, as a blocking join (parallel_invoke) does.
	// Prioritized steps run first, highest priority first: textures (short, DDS reads limited to the threads that saturate
	// read bandwidth), then meshes by index count so the longest chains start early. Steps on a mesh's critical path
	// (chains, their meshlets, the merge) get twice the priority of those off it (LOD 0 meshlets, shadow LOD 0 remap).
	void run_prepare_graph()
	{
		PROFILER_SCOPE();
		using step = tbb::flow::continue_node<tbb::flow::continue_msg>;
		using texture_step = tbb::flow::function_node<uint32_t, tbb::flow::continue_msg>;
		constexpr size_t dds_read_concurrency = 4;
		constexpr auto texture_priority = std::numeric_limits<tbb::flow::node_priority_t>::max();

		_mesh_upload_data.resize( _parsed_meshes.size() );
		_images.resize( _images_usage.size() );
		// Written by upload(), read by finalize()
		_meshes.resize( _asset.meshes.size() );
		_textures.resize( _textures_usage.size() );
		// Main LOD 0, main LODs 1+, shadow LOD 0, shadow LODs 1+: built in parallel, merged in that order
		std::vector<std::array<meshlet_set, 4>> meshlets( _parsed_meshes.size() );

		tbb::flow::graph graph;
		std::vector<std::unique_ptr<step>> steps;
		steps.reserve( _parsed_meshes.size() * 9 );
		const auto add_step = [ & ]( tbb::flow::node_priority_t priority, auto body ) -> step&
		{
			return *steps.emplace_back( std::make_unique<step>(
				graph,
				[ body = std::move( body ) ]( const tbb::flow::continue_msg& )
				{
					body();
					return tbb::flow::continue_msg { };
				},
				priority ) );
		};

		std::vector<step*> roots;
		roots.reserve( _parsed_meshes.size() );
		for ( uint32_t m = 0; m < _parsed_meshes.size(); ++m )
		{
			auto& data = _mesh_upload_data[ m ];
			auto& sets = meshlets[ m ];
			// 0 means no priority, and texture_priority must stay above
			const auto side = static_cast<tbb::flow::node_priority_t>(
								  std::min<size_t>( _parsed_meshes[ m ].indices.length, texture_priority / 4 ) )
				+ 1;
			const auto critical = 2 * side;

			auto& repack = add_step( critical, [ this, m ] { repack_mesh( m ); } );
			auto& main_lods = add_step( critical,
										[ &data ]
										{
											PROFILER_SCOPE_N( "lod_generation" );
											generate_lods( data.vertices, data.indices, data.lods, true );
										} );
			auto& shadow_lods = add_step( critical,
										  [ &data ]
										  {
											  PROFILER_SCOPE_N( "shadow_lod_generation" );
											  // Simplify from the original indices: merging vertices by position first creates
											  // non-manifold edges (eg: duplicated back faces) which stop the simplifier early.
											  // Attribute seams don't matter for depth-only rendering though, so every LOD is
											  // remapped afterwards to only distinguish vertices by position (fewer unique
											  // vertices to transform)
											  generate_lods( data.vertices, data.indices, data.shadow_lods, false );
											  size_t lod_offset = 0;
											  for ( const auto lod_index_count : data.shadow_lods.indices_per_lod )
											  {
												  remap_to_shadow_indices(
													  data.vertices,
													  std::span( data.shadow_lods.indices ).subspan( lod_offset, lod_index_count ) );
												  lod_offset += lod_index_count;
											  }
										  } );
			auto& shadow_lod0 = add_step( side,
										  [ &data ]
										  {
											  PROFILER_SCOPE_N( "shadow_lod0_remap" );
											  data.shadow_indices.assign( begin( data.indices ), end( data.indices ) );
											  remap_to_shadow_indices( data.vertices, data.shadow_indices );
										  } );
			auto& main_meshlets0 = add_step( side,
											 [ &data, &sets ]
											 {
												 PROFILER_SCOPE_N( "meshlet_generation" );
												 build_meshlets( data.vertices, data.indices, data.lods, 0, 0, sets[ 0 ] );
											 } );
			auto& main_meshlets = add_step( critical,
											[ &data, &sets ]
											{
												PROFILER_SCOPE_N( "meshlet_generation" );
												build_meshlets( data.vertices, data.indices, data.lods, 1, MESH_LOD_COUNT - 1, sets[ 1 ] );
											} );
			auto& shadow_meshlets0 = add_step( side,
											   [ &data, &sets ]
											   {
												   PROFILER_SCOPE_N( "meshlet_generation" );
												   build_meshlets( data.vertices, data.shadow_indices, data.shadow_lods, 0, 0, sets[ 2 ] );
											   } );
			auto& shadow_meshlets = add_step(
				critical,
				[ &data, &sets ]
				{
					PROFILER_SCOPE_N( "meshlet_generation" );
					build_meshlets( data.vertices, data.shadow_indices, data.shadow_lods, 1, MESH_LOD_COUNT - 1, sets[ 3 ] );
				} );
			auto& merge = add_step( critical,
									[ this, m, &data, &sets ]
									{
										PROFILER_SCOPE_N( "merge_meshlets" );
										merge_meshlets( data, sets );
										sets = { };
										_uploads.push( { upload_item::kind::MESH, this, m } );
									} );

			tbb::flow::make_edge( repack, main_lods );
			tbb::flow::make_edge( repack, shadow_lods );
			tbb::flow::make_edge( repack, shadow_lod0 );
			tbb::flow::make_edge( repack, main_meshlets0 );
			tbb::flow::make_edge( shadow_lod0, shadow_meshlets0 );
			tbb::flow::make_edge( main_lods, main_meshlets );
			tbb::flow::make_edge( shadow_lods, shadow_meshlets );
			tbb::flow::make_edge( main_meshlets0, merge );
			tbb::flow::make_edge( main_meshlets, merge );
			tbb::flow::make_edge( shadow_meshlets0, merge );
			tbb::flow::make_edge( shadow_meshlets, merge );
			roots.push_back( &repack );
		}

		// Both take a used texture's index: an image backs at most one texture (see add_texture_usage())
		texture_step dds_reads(
			graph,
			dds_read_concurrency,
			[ this ]( uint32_t texture )
			{
				read_dds( get_image_index( _asset.textures[ texture ] ) );
				_uploads.push( { upload_item::kind::TEXTURE, this, texture } );
				return tbb::flow::continue_msg { };
			},
			texture_priority );
		texture_step decodes(
			graph,
			tbb::flow::unlimited,
			[ this ]( uint32_t texture )
			{
				decode_image( get_image_index( _asset.textures[ texture ] ) );
				_uploads.push( { upload_item::kind::TEXTURE, this, texture } );
				return tbb::flow::continue_msg { };
			},
			texture_priority );

		for ( size_t i = 0; i < _textures_usage.size(); ++i )
		{
			if ( _textures_usage[ i ] != texture_usage::UNUSED )
			{
				const auto texture = static_cast<uint32_t>( i );
				const auto image = get_image_index( _asset.textures[ i ] );
				( _images_usage[ image ] == image_usage::DDS ? dds_reads : decodes ).try_put( texture );
			}
		}
		for ( const auto root : roots )
		{
			root->try_put( tbb::flow::continue_msg { } );
		}
		graph.wait_for_all();
	}

	// Expands 16-bit indices, generates missing tangents and packs the vertices
	void repack_mesh( uint32_t mesh_index )
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
	}

	// Simplify lod0_indices into the LODs 1+ of the chain, each LOD from the previous one as meshoptimizer recommends: the
	// collapses done for the previous LODs aren't redone, and with meshopt_SimplifySparse a coarse LOD's cost scales with
	// its own indices rather than with the whole vertex buffer
	// Attributes (normals) are only taken into account if use_attributes is set, which is meant for regular (shaded) rendering
	static void
	generate_lods( std::span<const mesh_vertex> vertices, std::span<const uint32_t> lod0_indices, lod_chain& chain, bool use_attributes )
	{
		// Each LOD is as coarse as its error limit allows (no index count target), limits double from one LOD to the next
		// so that the LOD selected for a pixel tolerance never exceeds twice the error it could afford
		constexpr float first_lod_error = 0.0025f;
		std::vector<uint32_t> buffer( lod0_indices.size() );
		// LOD selection compares errors against world space distances (in mesh units, before instance scale), so they're
		// absolute: limits are relative to the whole mesh extents, the scale the simplifier uses for relative errors
		const float error_scale = meshopt_simplifyScale( &vertices.front().position.x, vertices.size(), sizeof( mesh_vertex ) );
		std::span<const uint32_t> source = lod0_indices;
		// Each step's result error is measured against its source LOD, not LOD0. Like niagara (by meshoptimizer's author), the
		// LOD's error grows by at least 1.5x per step rather than summing the steps' errors, which is a strict bound but
		// inflates errors enough to select much denser LODs. Every step gets the LOD's full limit, so recorded errors stay
		// within it: 1.5x the previous limit is 0.75x this one.
		float source_error = 0.f;
		for ( size_t lod = 0; lod < chain.indices_per_lod.size(); ++lod )
		{
			const float target_error = first_lod_error * static_cast<float>( 1 << lod ) * error_scale;
			// Sparse only pays off on subsets, LOD0 references most of the vertices
			const unsigned int options = meshopt_SimplifyErrorAbsolute | ( lod > 0 ? meshopt_SimplifySparse : 0 );
			float step_error = 0.f;
			size_t size = 0;
			if ( use_attributes )
			{
				const float attr_weights[ 3 ] = { 0.5f, 0.5f, 0.5f };
				size = meshopt_simplifyWithAttributes( buffer.data(),
													   source.data(),
													   source.size(),
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
													   options,
													   &step_error );
			}
			else
			{
				size = meshopt_simplify( buffer.data(),
										 source.data(),
										 source.size(),
										 &vertices.front().position.x,
										 vertices.size(),
										 sizeof( mesh_vertex ),
										 0,
										 target_error,
										 options,
										 &step_error );
			}
			source_error = std::max( source_error * 1.5f, step_error );
			chain.error_per_lod[ lod ] = source_error;

			meshopt_optimizeVertexCache( buffer.data(), buffer.data(), size, vertices.size() );
			chain.indices_per_lod[ lod ] = static_cast<uint32_t>( size );
			const auto offset = chain.indices.size();
			chain.indices.insert( end( chain.indices ), buffer.data(), buffer.data() + size );
			source = std::span( chain.indices ).subspan( offset, size );
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

	// Build meshlets for LODs first_lod to last_lod of the chain (LOD 0 being lod0_indices) into a set, with offsets relative
	// to the set's own buffers. Only writes the chain's meshlets_per_lod for those LODs, so other LODs can be built at the
	// same time.
	static void build_meshlets( std::span<const mesh_vertex> vertices,
								std::span<const uint32_t> lod0_indices,
								lod_chain& chain,
								int first_lod,
								int last_lod,
								meshlet_set& out )
	{
		std::vector<meshopt_Meshlet> meshlet_buffer;
		size_t index_offset = 0;
		for ( int lod = 1; lod < first_lod; ++lod )
		{
			index_offset += chain.indices_per_lod[ lod - 1 ];
		}
		for ( int lod = first_lod; lod <= last_lod; ++lod )
		{
			const auto indices_ptr = lod == 0 ? lod0_indices.data() : chain.indices.data() + index_offset;
			const auto indices_count = lod == 0 ? lod0_indices.size() : chain.indices_per_lod[ lod - 1 ];
			index_offset += lod == 0 ? 0 : indices_count;

			if ( indices_count > 0 )
			{
				auto max_meshlets = meshopt_buildMeshletsBound( indices_count, 64, 96 );
				meshlet_buffer.resize( max_meshlets );
				const auto vertices_offset = out.vertices.size();
				out.vertices.resize( out.vertices.size() + indices_count );
				const auto triangles_offset = out.triangles.size();
				out.triangles.resize( out.triangles.size() + indices_count + max_meshlets * 3 );

				const auto meshlet_count = meshopt_buildMeshlets( meshlet_buffer.data(),
																  out.vertices.data() + vertices_offset,
																  out.triangles.data() + triangles_offset,
																  indices_ptr,
																  indices_count,
																  &vertices.front().position.x,
																  vertices.size(),
																  sizeof( mesh_vertex ),
																  64,
																  96,
																  0.f );

				chain.meshlets_per_lod[ lod ] = static_cast<uint32_t>( meshlet_count );
				const auto& last = meshlet_buffer[ meshlet_count - 1 ];

				out.vertices.resize( vertices_offset + last.vertex_offset + last.vertex_count );
				out.triangles.resize( triangles_offset + last.triangle_offset + last.triangle_count * 3 );

				for ( const auto& meshlet : std::span( meshlet_buffer.data(), meshlet_count ) )
				{
					const size_t vertex_index = meshlet.vertex_offset + vertices_offset;
					const size_t triangle_index = meshlet.triangle_offset + triangles_offset;
					const auto bounds = meshopt_computeMeshletBounds( out.vertices.data() + vertex_index,
																	  out.triangles.data() + triangle_index,
																	  meshlet.triangle_count,
																	  &vertices.front().position.x,
																	  vertices.size(),
																	  sizeof( mesh_vertex ) );

					out.meshlets.emplace_back(
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

	// Append the sets in order to the mesh meshlets, offsetting their vertex and triangle indices
	static void merge_meshlets( mesh_upload_data& upload_data, std::span<const meshlet_set> sets )
	{
		size_t meshlets = 0;
		size_t vertices = 0;
		size_t triangles = 0;
		for ( const auto& set : sets )
		{
			meshlets += set.meshlets.size();
			vertices += set.vertices.size();
			triangles += set.triangles.size();
		}
		upload_data.meshlets.reserve( meshlets );
		upload_data.meshlet_vertices.reserve( vertices );
		upload_data.meshlet_triangles.reserve( triangles );

		for ( const auto& set : sets )
		{
			const auto vertex_base = static_cast<uint32_t>( upload_data.meshlet_vertices.size() );
			const auto triangle_base = static_cast<uint32_t>( upload_data.meshlet_triangles.size() );
			upload_data.meshlet_vertices.insert( end( upload_data.meshlet_vertices ), begin( set.vertices ), end( set.vertices ) );
			upload_data.meshlet_triangles.insert( end( upload_data.meshlet_triangles ), begin( set.triangles ), end( set.triangles ) );
			for ( auto meshlet : set.meshlets )
			{
				meshlet.vertex_offset += vertex_base;
				meshlet.triangle_offset += triangle_base;
				upload_data.meshlets.push_back( meshlet );
			}
		}
	}

	void upload_mesh( gpu_state& gpu, size_t mesh_index )
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
		mesh.meshlets_count = upload_lods( gpu, upload_data.indices, upload_data.lods, 0, mesh.gpu_data.lods );
		upload_lods( gpu, upload_data.shadow_indices, upload_data.shadow_lods, mesh.meshlets_count, mesh.gpu_data.shadow_lods );
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

		mesh.data_buffer = gpu.device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
														 | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
													 data_buffer_size );

		mesh.gpu_data.vertices = mesh.data_buffer.get_device_address();
		gpu.mesh_upload.queue( upload_data.vertices, mesh.data_buffer );
		auto offset = upload_data.vertices.size() * sizeof( mesh_vertex );

		mesh.gpu_data.meshlets = mesh.data_buffer.get_device_address() + offset;
		gpu.mesh_upload.queue( upload_data.meshlets, mesh.data_buffer, offset );
		offset += upload_data.meshlets.size() * sizeof( gpu_meshlet );

		mesh.gpu_data.meshlet_vertices = mesh.data_buffer.get_device_address() + offset;
		gpu.mesh_upload.queue( upload_data.meshlet_vertices, mesh.data_buffer, offset );
		offset += upload_data.meshlet_vertices.size() * sizeof( uint32_t );

		mesh.gpu_data.meshlet_triangles = mesh.data_buffer.get_device_address() + offset;
		gpu.mesh_upload.queue( upload_data.meshlet_triangles, mesh.data_buffer, offset );
		offset += upload_data.meshlet_triangles.size() * sizeof( uint8_t );
	}

	// Upload the chain indices (LOD 0 first) to the index buffer and fill the matching LOD descriptors
	// Meshlets of the chain are expected to be contiguous, starting at first_meshlet
	// Returns the meshlet offset following the chain
	uint32_t upload_lods( gpu_state& gpu,
						  std::span<const uint32_t> lod0_indices,
						  const lod_chain& chain,
						  uint32_t first_meshlet,
						  std::array<gpu_mesh_lod, MESH_LOD_COUNT>& lods )
	{
		const size_t index_count = lod0_indices.size() + chain.indices.size();
		auto index_offset = static_cast<uint32_t>( gpu.indices.size.fetch_add( index_count ) );
		if ( index_offset + index_count > gpu.indices.capacity )
		{
			throw renderer::Error( "Index buffer is full" );
		}
		gpu.mesh_upload.queue( lod0_indices, gpu.indices.buffer, index_offset * sizeof( uint32_t ) );
		if ( !chain.indices.empty() )
		{
			gpu.mesh_upload.queue( chain.indices, gpu.indices.buffer, ( index_offset + lod0_indices.size() ) * sizeof( uint32_t ) );
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

	// The file is kept as is, parsed at upload once its sRGB usage is known
	void read_dds( size_t index )
	{
		PROFILER_SCOPE();
		_images[ index ] = load( _asset.images[ index ].data, true );
	}

	// PNG or JPEG, decoded to RGBA32
	void decode_image( size_t index )
	{
		PROFILER_SCOPE();
		const auto source = load( _asset.images[ index ].data, true );
		PROFILER_SCOPE_PUSH( decompress_png, "decompress_png" );
		sdl::raii::Surface surface( IMG_Load_IO( SDL_IOFromConstMem( source.data.data(), source.data.size() ), true ) );
		PROFILER_SCOPE_POP( decompress_png );
		if ( !surface )
		{
			throw renderer::Error( SDL_GetError() );
		}
		// Every format is expanded to RGBA, including palettes and grayscale
		else if ( surface->format != SDL_PIXELFORMAT_RGBA32 )
		{
			PROFILER_SCOPE_N( "convert_to_rgba" );
			sdl::raii::Surface converted_image( SDL_ConvertSurface( surface.get(), SDL_PIXELFORMAT_RGBA32 ) );
			if ( !converted_image )
			{
				throw renderer::Error( SDL_GetError() );
			}
			surface = std::move( converted_image );
		}
		_images[ index ] = std::move( surface );
	}

	// Texture i must be used, and its image decoded
	void upload_texture( gpu_state& gpu, size_t i )
	{
		PROFILER_SCOPE();
		const auto usage = _textures_usage[ i ];
		const auto& image = _images[ get_image_index( _asset.textures[ i ] ) ];
		if ( const auto dds_source = std::get_if<data_source>( &image ) )
		{
			const DDSFile dds( dds_source->data, is_srgb( usage ) );
			auto tex = make_image_from_dds( gpu.device, dds );
			gpu.texture_upload.queue( dds, tex );
			_textures[ i ] = gpu.bindless_manager.add_texture( std::move( tex ) );
		}
		else if ( const auto& surface = std::get<sdl::raii::Surface>( image ); gpu.compressor )
		{
			auto tex = make_image_from_surface( gpu.device, *surface, compressed_format( usage ) );
			gpu.texture_upload.queue( *surface, tex, *gpu.compressor );
			_textures[ i ] = gpu.bindless_manager.add_texture( std::move( tex ) );
		}
		else
		{
			auto tex = make_image_from_surface( gpu.device, *surface, uncompressed_format( usage ) );
			gpu.texture_upload.queue( *surface, tex );
			_textures[ i ] = gpu.bindless_manager.add_texture( std::move( tex ) );
		}
	}

	loaded_asset finalize_impl( BindlessManager& bindless_manager )
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

		_materials.resize( _asset.materials.size() );
		for ( size_t i = 0; i < _materials_usage.size(); ++i )
		{
			if ( _materials_usage[ i ] )
			{
				auto& material = _materials[ i ];
				if ( _asset.materials[ i ].normalTexture )
				{
					material.gpu_data.normal_texture = _textures[ _asset.materials[ i ].normalTexture->textureIndex ]
														   .handles.sampled_texture;
				}

				// parse() checked that a material uses only one workflow
				if ( const auto& specular_glossiness = _asset.materials[ i ].specularGlossiness )
				{
					material.gpu_data.workflow = material_workflow::SPECULAR_GLOSSINESS;
					if ( specular_glossiness->diffuseTexture )
					{
						material.gpu_data.albedo_texture = _textures[ specular_glossiness->diffuseTexture->textureIndex ]
															   .handles.sampled_texture;
					}
					if ( specular_glossiness->specularGlossinessTexture )
					{
						material.gpu_data
							.specular_glossiness_texture = _textures[ specular_glossiness->specularGlossinessTexture->textureIndex ]
															   .handles.sampled_texture;
					}
					material.gpu_data.base_color = make_vec4( specular_glossiness->diffuseFactor );
					material.gpu_data.specular_glossiness = make_vec4( specular_glossiness->specularFactor,
																	   specular_glossiness->glossinessFactor );
				}
				else
				{
					if ( _asset.materials[ i ].pbrData.baseColorTexture )
					{
						material.gpu_data.albedo_texture = _textures[ _asset.materials[ i ].pbrData.baseColorTexture->textureIndex ]
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
					material.gpu_data.base_color = make_vec4( _asset.materials[ i ].pbrData.baseColorFactor );
					material.gpu_data.metallic = _asset.materials[ i ].pbrData.metallicFactor;
					material.gpu_data.roughness = _asset.materials[ i ].pbrData.roughnessFactor;
				}

				if ( _asset.materials[ i ].occlusionTexture )
				{
					material.gpu_data.occlusion_strength = _asset.materials[ i ].occlusionTexture->strength;
					material.gpu_data.occlusion_texture = _textures[ _asset.materials[ i ].occlusionTexture->textureIndex ]
															  .handles.sampled_texture;
				}

				material.gpu_data.emissive = make_vec3( _asset.materials[ i ].emissiveFactor ) * _asset.materials[ i ].emissiveStrength;
				if ( _asset.materials[ i ].emissiveTexture )
				{
					material.gpu_data.emissive_texture = _textures[ _asset.materials[ i ].emissiveTexture->textureIndex ]
															 .handles.sampled_texture;
				}

				switch ( _asset.materials[ i ].alphaMode )
				{

					case fastgltf::AlphaMode::Opaque:
						material.gpu_data.type = _asset.materials[ i ].doubleSided ? material_type::OPAQUE_DOUBLE
																				   : material_type::OPAQUE_SINGLE;
						break;
					case fastgltf::AlphaMode::Mask:
						material.gpu_data.type = _asset.materials[ i ].doubleSided ? material_type::MASK_DOUBLE
																				   : material_type::MASK_SINGLE;
						material.gpu_data.alpha_cutoff = _asset.materials[ i ].alphaCutoff;
						break;
					case fastgltf::AlphaMode::Blend:
						material.gpu_data.type = _asset.materials[ i ].doubleSided ? material_type::ALPHA_DOUBLE
																				   : material_type::ALPHA_SINGLE;
						// glTF ignores alphaCutoff when blending, this is the depth writing pass's own threshold
						material.gpu_data.alpha_cutoff = 0.5f;
						break;
					default:
						throw renderer::Error( "Unsupported material alpha mode" );
				}
				material.type = material.gpu_data.type;

				material.handle = bindless_manager.add_buffer_entry( material.gpu_data );
			}
		}

		for ( const auto& parsed_mesh : _parsed_meshes )
		{
			auto& mesh = _meshes[ parsed_mesh.gltf_index ];
			mesh.handle = bindless_manager.add_buffer_entry( mesh.gpu_data );
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
	bool _use_dds;
	std::filesystem::path _path;
	fastgltf::Asset _asset;
	std::vector<uint8_t> _meshes_usage; // Could be a dynamic bitset
	std::vector<image_usage> _images_usage;
	std::vector<texture_usage> _textures_usage;
	std::vector<uint8_t> _materials_usage; // Could be a dynamic bitset
	std::vector<mesh_data> _parsed_meshes;
	std::vector<mesh_upload_data> _mesh_upload_data;
	upload_queue& _uploads;
	std::exception_ptr _error;
	std::vector<data_source> _buffers;
	std::vector<loaded_image> _images;
	std::vector<renderer::BindlessTexture> _textures;
	std::vector<material> _materials;
	std::vector<mesh> _meshes;
};

Assets::Assets( size_t max_indices, const texture_options& options )
	: _max_indices( max_indices )
	, _texture_options( options )
{
}

Assets::~Assets()
{
	// Loads that were never waited for, after an error elsewhere: their jobs must outlive their tasks and uploads
	_prepare_tasks.cancel();
	try
	{
		_prepare_tasks.wait();
	}
	catch ( ... )
	{
	}
}

scene_handle Assets::load( std::filesystem::path path )
{
	PROFILER_SCOPE();
	const scene_handle handle { static_cast<uint32_t>( _scenes.size() + _jobs.size() ) };
	const auto job = _jobs.emplace_back( std::make_unique<load_job>( _texture_options.use_dds, std::move( path ), _uploads ) ).get();
	tbb::this_task_arena::enqueue( [ job ] { job->prepare(); }, _prepare_tasks );
	return handle;
}

void Assets::attach( renderer::Device& device, BindlessManager& bindless_manager, texture_compressor* compressor )
{
	PROFILER_SCOPE();
	assert( !_gpu );
	assert( !compressor || compressor->is_ready() );
	_gpu.reset( new gpu_state { .device = device,
								.bindless_manager = bindless_manager,
								.compressor = compressor,
								.indices = index_buffer( device, _max_indices ) } );
}

// Runs on the calling thread instead of joining the TBB workers in _prepare_tasks.wait(), so it doesn't oversubscribe
// the cores. The only user of the device's command pool and queue until it returns: the pipeline manager's thread only
// creates pipelines.
void Assets::upload_prepared()
{
	PROFILER_SCOPE();
	upload_state gpu { .device = _gpu->device,
					   .bindless_manager = _gpu->bindless_manager,
					   .indices = _gpu->indices,
					   .compressor = _gpu->compressor,
					   .mesh_upload = gpu_upload_job( _gpu->device, 32 * 1024 * 1024 ),
					   .texture_upload = gpu_upload_job( _gpu->device, 128 * 1024 * 1024, _gpu->compressor ? 64 * 1024 * 1024 : 0 ) };
	size_t pending_jobs = _jobs.size();
	upload_item item;
	while ( pending_jobs != 0 )
	{
		if ( !_uploads.try_pop( item ) )
		{
			// Nothing ready: get the GPU copying what was queued while waiting for more
			{
				PROFILER_SCOPE_N( "flush_uploads" );
				gpu.mesh_upload.flush();
				gpu.texture_upload.flush();
			}
			PROFILER_SCOPE_N( "wait_prepared" );
			_uploads.pop( item );
		}
		if ( item.kind == upload_item::kind::JOB_DONE )
		{
			--pending_jobs;
		}
		else
		{
			item.job->upload( gpu, item );
		}
	}
	{
		PROFILER_SCOPE_N( "finish_uploads" );
		gpu.mesh_upload.submit();
		gpu.texture_upload.submit();
		gpu_upload_job::wait_all_done( { &gpu.mesh_upload, &gpu.texture_upload } );
	}
}

void Assets::wait_ready()
{
	PROFILER_SCOPE();
	assert( _gpu );
	upload_prepared();
	// Every job pushed JOB_DONE, this only waits for their tasks to return
	_prepare_tasks.wait();
	for ( const auto& job : _jobs )
	{
		if ( job->get_error() )
		{
			std::rethrow_exception( job->get_error() );
		}
	}
	for ( auto& job : _jobs )
	{
		auto asset = job->finalize( _gpu->bindless_manager );
		add_content( asset );
		_scenes.push_back( std::move( asset.contents ) );
		// Frees what's left of its CPU side data
		job.reset();
	}
	_jobs.clear();
}

const scene& Assets::get( scene_handle handle ) const
{
	assert( handle.index < _scenes.size() );
	return _scenes[ handle.index ];
}

const index_buffer& Assets::get_indices() const
{
	assert( _gpu );
	return _gpu->indices;
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
