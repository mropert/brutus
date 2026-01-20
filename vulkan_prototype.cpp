#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>
#include <array>
#include <chrono>
#include <fastgltf/core.hpp>
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
#include <numeric>
#include <optick.h>
#include <span>
#include <tbb/tbb.h>

//

#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/command_buffer.h>
#include <renderer/device.h>
#include <renderer/pipeline_manager.h>
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
	uint32_t albedo_texture_index;
	uint32_t normal_texture_index;
};

struct gpu_mesh_lod
{
	uint32_t indices_offset;
	uint32_t indices_count;
	uint32_t meshlets_offset;
	uint32_t meshlets_count;
};

struct gpu_meshlet
{
	uint32_t vertex_offset;
	uint32_t triangle_offset;
	uint32_t vertex_count;
	uint32_t triangle_count;
	glm::aligned_vec4 bounding_sphere; // z is radius
};

struct gpu_mesh
{
	glm::aligned_vec4 bounding_sphere; // z is radius
	std::array<gpu_mesh_lod, 4> lods;
	renderer::BufferHandle vertices;
	renderer::BufferHandle meshlets;
	renderer::BufferHandle meshlet_vertices;
	renderer::BufferHandle meshlet_triangles;
};

struct index_buffer
{
	index_buffer( renderer::Device& device, uint32_t max_size )
		: buffer( device.create_buffer( renderer::Buffer::Usage::INDEX_BUFFER | renderer::Buffer::Usage::TRANSFER_DST,
										max_size * sizeof( uint32_t ) ) )
		, capacity( max_size )
	{
	}

	renderer::raii::Buffer buffer;
	uint32_t size = 0;
	uint32_t capacity;
};

using mesh_handle = renderer::BindlessHandle<gpu_mesh>;
using material_handle = renderer::BindlessHandle<gpu_material>;

struct mesh
{
	renderer::raii::Buffer data_buffer; // vertex + meshlets
	mesh_handle handle;
	// Data below is only for debugging purposes, a CPU side copy of the GPU-side descriptor
	gpu_mesh gpu_data;
};

struct material
{
	material_handle handle;
	// Data below is only for debugging purposes, a CPU side copy of the GPU-side descriptor
	gpu_material gpu_data;
};

struct model
{
	std::vector<mesh> meshes;
	std::vector<material> materials;
	glm::vec4 bounding_sphere;
};

struct push_constants
{
	mesh_handle mesh;
	material_handle material;
	renderer::BufferHandle scene_constants;
	renderer::BufferHandle instance_buffer;
	float lod_cutoff = 10.;
};

using BindlessManager = renderer::BindlessManager<gpu_mesh, gpu_material>;
static constexpr uint32_t MAX_MESHES = 8192;
static constexpr uint32_t MAX_MATERIALS = 8192;

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
		assert( _offset + size < _size );
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
		assert( _offset + pitch * surface.h < _size );
		SDL_ConvertPixels( surface.w, surface.h, surface.format, surface.pixels, surface.pitch, gpu_format, ptr + _offset, pitch );
		_cmd_buffer->transition_texture( texture, renderer::Texture::Layout::UNDEFINED, renderer::Texture::Layout::TRANSFER_DST_OPTIMAL );
		_cmd_buffer->copy_buffer_to_texture( _upload_buffer, _offset, texture );
		_cmd_buffer->transition_texture( texture,
										 renderer::Texture::Layout::TRANSFER_DST_OPTIMAL,
										 renderer::Texture::Layout::SHADER_READ_ONLY_OPTIMAL );
		_offset += pitch * surface.h;
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
		assert( _state == state::UPLOADING );
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
	uint32_t _size;
	renderer::Device& _device;
	renderer::raii::CommandBuffer _cmd_buffer;
	renderer::raii::Buffer _upload_buffer;
	renderer::raii::Fence _upload_fence;
	uint32_t _offset = 0;
	state _state = state::UPLOADING;
};

class mesh_load_job
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
		pbr_data pbr_data;
	};


public:
	mesh_load_job( renderer::Device& device, BindlessManager& bindless_manager, index_buffer& index_buffer, std::filesystem::path path )
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

		_meshes.resize( _parsed_meshes.size() );
		_materials.resize( _parsed_meshes.size() );
	}

	struct lod_indices
	{
		std::vector<uint32_t> indices;
		std::array<uint32_t, 3> indices_per_lod;
	};

	void upload_mesh()
	{
		OPTICK_EVENT();
		const auto& gltf_buffer = std::get<fastgltf::sources::Array>( _asset.buffers[ 0 ].data ).bytes;

		std::vector<std::vector<mesh_vertex>> vertex_data( _meshes.size() );
		for ( int i = 0; i < _meshes.size(); ++i )
		{
			OPTICK_EVENT( "repack_vertices" );
			vertex_data[ i ].reserve( _parsed_meshes[ i ].positions.length );

			const auto positions = reinterpret_cast<const float*>( gltf_buffer.data() + _parsed_meshes[ i ].positions.offset );
			const auto normals = reinterpret_cast<const float*>( gltf_buffer.data() + _parsed_meshes[ i ].normals.offset );
			const auto tangents = reinterpret_cast<const float*>( gltf_buffer.data() + _parsed_meshes[ i ].tangents.offset );
			const auto uvs = reinterpret_cast<const float*>( gltf_buffer.data() + _parsed_meshes[ i ].uvs.offset );
			for ( int v = 0; v < _parsed_meshes[ i ].positions.length; ++v )
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
				buffer.resize( _parsed_meshes[ i ].indices.length );
				const auto indices = reinterpret_cast<const uint32_t*>( gltf_buffer.data() + _parsed_meshes[ i ].indices.offset );
				for ( int lod = 0; lod < simplified_indices[ i ].indices_per_lod.size(); ++lod )
				{
					const float attr_weights[ 3 ] = { 0.5f, 0.5f, 0.5f };
					const uint32_t size = meshopt_simplifyWithAttributes( buffer.data(),
																		  indices,
																		  _parsed_meshes[ i ].indices.length,
																		  &vertex_data[ i ].front().position.x,
																		  vertex_data[ i ].size(),
																		  sizeof( mesh_vertex ),
																		  &vertex_data[ i ].front().normal.x,
																		  sizeof( mesh_vertex ),
																		  attr_weights,
																		  3,
																		  nullptr,
																		  _parsed_meshes[ i ].indices.length * lod_indices_factor[ lod ],
																		  lod_error_factor[ lod ],
																		  0,
																		  nullptr );

					meshopt_optimizeVertexCache( buffer.data(), buffer.data(), size, vertex_data[ i ].size() );
					simplified_indices[ i ].indices_per_lod[ lod ] = size;
					simplified_indices[ i ].indices.insert( end( simplified_indices[ i ].indices ), buffer.data(), buffer.data() + size );
				}
			}
		}

		std::vector<std::vector<gpu_meshlet>> all_meshlets;
		std::vector<std::vector<uint32_t>> all_meshlet_vertices;
		std::vector<std::vector<uint8_t>> all_meshlet_triangles;
		{
			OPTICK_EVENT( "meshlet_generation" );
			std::vector<meshopt_Meshlet> meshlet_buffer;
			for ( int i = 0; i < _meshes.size(); ++i )
			{
				uint32_t index_offset = 0;
				uint32_t meshlet_vertices_offset = 0;
				uint32_t meshlet_triangles_offset = 0;
				for ( int lod = 0; lod < 4; ++lod )
				{
					const auto indices = lod == 0
						? reinterpret_cast<const uint32_t*>( gltf_buffer.data() + _parsed_meshes[ i ].indices.offset )
						: simplified_indices[ i ].indices.data() + index_offset;
					const auto indices_count = lod == 0 ? _parsed_meshes[ i ].indices.length
														: simplified_indices[ i ].indices_per_lod[ lod - 1 ];
					index_offset += lod == 0 ? 0 : indices_count;

					auto max_meshlets = meshopt_buildMeshletsBound( indices_count, 64, 96 );
					meshlet_buffer.resize( max_meshlets );
					auto& meshlet_vertices = all_meshlet_vertices.emplace_back( indices_count );
					auto& meshlet_triangles = all_meshlet_triangles.emplace_back( indices_count + max_meshlets * 3 );

					const auto meshlet_count = meshopt_buildMeshlets( meshlet_buffer.data(),
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

					const auto& last = meshlet_buffer[ meshlet_count - 1 ];

					meshlet_vertices.resize( last.vertex_offset + last.vertex_count );
					meshlet_triangles.resize( last.triangle_offset + last.triangle_count * 3 );

					auto& meshlets = all_meshlets.emplace_back();
					meshlets.reserve( meshlet_count );

					for ( const auto& meshlet : std::span( meshlet_buffer.data(), meshlet_count ) )
					{
						const auto bounds = meshopt_computeMeshletBounds( meshlet_vertices.data() + meshlet.vertex_offset,
																		  meshlet_triangles.data() + meshlet.triangle_offset,
																		  meshlet.triangle_count,
																		  &vertex_data[ i ].front().position.x,
																		  vertex_data[ i ].size(),
																		  sizeof( mesh_vertex ) );

						// Meshes store meshlets vertices and triangles for all LODs together in one array so we need to fixup the base offset
						// We could also store the base offset in each LOD definition but that wouldn't save us any space
						meshlets.emplace_back(
							meshlet.vertex_offset + meshlet_vertices_offset,
							meshlet.triangle_offset + meshlet_triangles_offset,
							meshlet.vertex_count,
							meshlet.triangle_count,
							glm::aligned_vec4( bounds.center[ 0 ], bounds.center[ 1 ], bounds.center[ 2 ], bounds.radius ) );
					}
					meshlet_vertices_offset += meshlet_vertices.size();
					meshlet_triangles_offset += meshlet_triangles.size();
				}
			}
		}

		for ( int i = 0; i < _parsed_meshes.size(); ++i )
		{
			const auto& parsed_mesh = _parsed_meshes[ i ];
			auto& mesh = _meshes[ i ];

			const auto bounds = meshopt_computeSphereBounds( &vertex_data[ i ][ 0 ].position.x,
															 vertex_data[ i ].size(),
															 sizeof( mesh_vertex ),
															 nullptr,
															 0 );
			mesh.gpu_data.bounding_sphere.x = bounds.center[ 0 ];
			mesh.gpu_data.bounding_sphere.y = bounds.center[ 1 ];
			mesh.gpu_data.bounding_sphere.z = bounds.center[ 2 ];
			mesh.gpu_data.bounding_sphere.w = bounds.radius;

			// Index buffer
			_mesh_upload.queue( gltf_buffer.data() + parsed_mesh.indices.offset,
								parsed_mesh.indices.length * sizeof( uint32_t ),
								_index_buffer.buffer,
								_index_buffer.size * sizeof( uint32_t ) );
			mesh.gpu_data.lods[ 0 ].indices_offset = _index_buffer.size;
			mesh.gpu_data.lods[ 0 ].indices_count = parsed_mesh.indices.length;
			_index_buffer.size += parsed_mesh.indices.length;

			_mesh_upload.queue( simplified_indices[ i ].indices, _index_buffer.buffer, _index_buffer.size * sizeof( uint32_t ) );
			for ( int lod = 1; lod < mesh.gpu_data.lods.size(); ++lod )
			{
				mesh.gpu_data.lods[ lod ].indices_offset = mesh.gpu_data.lods[ lod - 1 ].indices_offset
					+ mesh.gpu_data.lods[ lod - 1 ].indices_count;
				mesh.gpu_data.lods[ lod ].indices_count = simplified_indices[ i ].indices_per_lod[ lod - 1 ];
			}
			_index_buffer.size += simplified_indices[ i ].indices.size();

			// Data buffer (vertices + meshlets)
			auto data_buffer_size = vertex_data[ i ].size() * sizeof( mesh_vertex );
			for ( const auto& meshlets : std::span( all_meshlets.data() + ( i * 4 ), 4 ) )
			{
				data_buffer_size += meshlets.size() * sizeof( gpu_meshlet );
			}
			for ( const auto& meshlet_vertices : std::span( all_meshlet_vertices.data() + ( i * 4 ), 4 ) )
			{
				data_buffer_size += meshlet_vertices.size() * sizeof( uint32_t );
			}
			for ( const auto& meshlet_triangles : std::span( all_meshlet_triangles.data() + ( i * 4 ), 4 ) )
			{
				data_buffer_size += meshlet_triangles.size() * sizeof( uint8_t );
			}

			mesh.data_buffer = _device.create_buffer( renderer::Buffer::Usage::STORAGE_BUFFER | renderer::Buffer::Usage::TRANSFER_DST
														  | renderer::Buffer::Usage::SHADER_DEVICE_ADDRESS,
													  data_buffer_size );

			mesh.gpu_data.vertices = mesh.data_buffer.get_device_address();
			_mesh_upload.queue( vertex_data[ i ], mesh.data_buffer );
			auto offset = vertex_data[ i ].size() * sizeof( mesh_vertex );

			mesh.gpu_data.meshlets = mesh.data_buffer.get_device_address() + offset;
			for ( int lod = 0; lod < 4; ++lod )
			{
				_mesh_upload.queue( all_meshlets[ i * 4 + lod ], mesh.data_buffer, offset );
				offset += all_meshlets[ i * 4 + lod ].size() * sizeof( gpu_meshlet );
				mesh.gpu_data.lods[ lod ].meshlets_count = all_meshlets[ i * 4 + lod ].size();
				if ( lod > 0 )
				{
					mesh.gpu_data.lods[ lod ].meshlets_offset = mesh.gpu_data.lods[ lod - 1 ].meshlets_offset
						+ mesh.gpu_data.lods[ lod - 1 ].meshlets_count;
				}
			}

			mesh.gpu_data.meshlet_vertices = mesh.data_buffer.get_device_address() + offset;
			for ( int lod = 0; lod < 4; ++lod )
			{
				_mesh_upload.queue( all_meshlet_vertices[ i * 4 + lod ], mesh.data_buffer, offset );
				offset += all_meshlet_vertices[ i * 4 + lod ].size() * sizeof( uint32_t );
			}

			mesh.gpu_data.meshlet_triangles = mesh.data_buffer.get_device_address() + offset;
			for ( int lod = 0; lod < 4; ++lod )
			{
				_mesh_upload.queue( all_meshlet_triangles[ i * 4 + lod ], mesh.data_buffer, offset );
				offset += all_meshlet_triangles[ i * 4 + lod ].size() * sizeof( uint8_t );
			}

			mesh.handle = _bindless_manager.add_buffer_entry( mesh.gpu_data );
		}

		_mesh_upload.submit();
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

			tex = make_image_from_surface( _device, *_textures[ parsed_mesh.normal_idx ].image, false );
			_texture_upload.queue( *_textures[ parsed_mesh.normal_idx ].image, tex );
			material.gpu_data.normal_texture_index = _bindless_manager.add_texture( std::move( tex ) ).handles.texture_index;

			material.gpu_data.pbr_data = parsed_mesh.pbr_data;

			material.handle = _bindless_manager.add_buffer_entry( material.gpu_data );
		}

		_texture_upload.submit();
	}

	model finalize()
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
		return model { std::move( _meshes ),
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

model load_model( renderer::Device& device, BindlessManager& bindless_manager, index_buffer& index_buffer )
{
	OPTICK_EVENT();
	mesh_load_job job( device, bindless_manager, index_buffer, "../assets/player.glb" );
	job.parse();
	tbb::parallel_invoke( [ & ]() { job.upload_mesh(); },
						  [ & ]
						  {
							  job.decompress_textures();
							  job.upload_materials();
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
	std::array<glm::aligned_vec4, 6> frustum;
	glm::aligned_vec3 camera_position;
	glm::aligned_vec4 main_light;
};

glm::vec4 normalize_plane( const glm::vec4& plane )
{
	return plane / glm::length( glm::vec3( plane ) );
}

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
	scene_constants constants { .view_projection_matrix = projection * view,
								.view_matrix = view,
								.inverse_view_matrix = glm::inverse( view ),
								.camera_position = camera.position,
								.main_light = main_light };

	// Math from https://github.com/zeux/niagara
	const float draw_distance = 500.f;
	const auto transposed_vp = glm::transpose( constants.view_projection_matrix );
	constants.frustum[ 0 ] = normalize_plane( transposed_vp[ 3 ] + transposed_vp[ 0 ] ); // x + w < 0
	constants.frustum[ 1 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 0 ] ); // x - w > 0
	constants.frustum[ 2 ] = normalize_plane( transposed_vp[ 3 ] + transposed_vp[ 1 ] ); // y + w < 0
	constants.frustum[ 3 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 1 ] ); // y - w > 0
	constants.frustum[ 4 ] = normalize_plane( transposed_vp[ 3 ] - transposed_vp[ 2 ] ); // z - w > 0 -- reverse z
	constants.frustum[ 5 ] = glm::vec4( 0, 0, -1, draw_distance ); // reverse z, infinite far plane

	return constants;
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

namespace Timestamps
{
	enum : uint32_t
	{
		FRAME_START = 0,
		DRAW_MESHES,
		DRAW_GRID = DRAW_MESHES + 64,
		DEPTH_PYRAMID,
		DRAW_UI,
		BLIT,
		FRAME_END,
		COUNT
	};
}

void draw_ui( ui_data& data,
			  const renderer::Device::Properties& props,
			  const BindlessManager& bindless_manager,
			  const index_buffer& index_buffer,
			  std::span<const mesh> meshes,
			  std::span<const uint64_t> gpu_timestamps,
			  float gpu_timestamp_period )
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
		ImGui::SeparatorText( "VRAM usage" );
		ImGui::Text( "Textures (MB): %zu", bindless_manager.get_texture_memory_usage() >> 20 );
		size_t meshes_bytes = 0;
		for ( const auto& mesh : meshes )
		{
			meshes_bytes += mesh.data_buffer.get_size();
		}
		ImGui::Text( "Meshes (MB): %zu", meshes_bytes >> 20 );
		ImGui::Text( "Indices (MB): %zu", ( index_buffer.size * sizeof( uint32_t ) ) >> 20 );

		std::array<float, 4> per_lod_gpu_time {};
		const auto gpu_time = ( gpu_timestamps[ Timestamps::FRAME_END ] - gpu_timestamps[ Timestamps::FRAME_START ] ) * gpu_timestamp_period
			* 1e-6f;
		const auto meshes_time = ( gpu_timestamps[ Timestamps::DRAW_GRID ] - gpu_timestamps[ Timestamps::DRAW_MESHES ] )
			* gpu_timestamp_period * 1e-6f;
		const auto depth_pyramid_time = ( gpu_timestamps[ Timestamps::DEPTH_PYRAMID + 1 ] - gpu_timestamps[ Timestamps::DEPTH_PYRAMID ] )
			* gpu_timestamp_period * 1e-6f;
		const auto ui_time = ( gpu_timestamps[ Timestamps::DRAW_UI + 1 ] - gpu_timestamps[ Timestamps::DRAW_UI ] ) * gpu_timestamp_period
			* 1e-6f;

		ImGui::SeparatorText( "Stats" );
		ImGui::Text( "GPU time (ms): %.2f", gpu_time );
		ImGui::BulletText( "Meshes: %.2f", meshes_time );
		ImGui::BulletText( "Depth pyramid: %.2f", depth_pyramid_time );
		ImGui::BulletText( "UI: %.2f", ui_time );
		for ( int i = 0; i < meshes.size(); ++i )
		{
			ImGui::Text( "Mesh #%d: %d / %d / %d / %d triangles",
						 i,
						 meshes[ i ].gpu_data.lods[ 0 ].indices_count,
						 meshes[ i ].gpu_data.lods[ 1 ].indices_count,
						 meshes[ i ].gpu_data.lods[ 2 ].indices_count,
						 meshes[ i ].gpu_data.lods[ 3 ].indices_count );
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

struct depth_pyramid_constants
{
	uint32_t depth_in;
	uint32_t depth_out;
	glm::aligned_vec2 size;
};

uint32_t calc_group_count( uint32_t count )
{
	return ( count + 32 - 1 ) / 32;
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
										 .DescriptorPoolSize = renderer::MAX_FRAMES_IN_FLIGHT,
										 .UseDynamicRendering = true,
										 .PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
																		  .colorAttachmentCount = 1,
																		  .pColorAttachmentFormats = &color_format } };
		ImGui_ImplVulkan_Init( &info );
	}

	const auto draw_image_extent = device.get_extent();
	using usage = renderer::Texture::Usage;
	auto draw_image = device.create_texture( renderer::Texture::Desc { .format = draw_image_format,
																	   .usage = usage::COLOR_ATTACHMENT,
																	   .extent = draw_image_extent,
																	   .samples = 4 } );
	auto draw_image_view = device.create_texture_view( draw_image, renderer::TextureView::Aspect::COLOR );

	auto depth_image = device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::D32_SFLOAT,
																		.usage = usage::DEPTH_STENCIL_ATTACHMENT,
																		.extent = draw_image_extent,
																		.samples = 4 } );

	auto depth_image_view = device.create_texture_view( depth_image, renderer::TextureView::Aspect::DEPTH );

	auto draw_resolved_image = device.create_texture( renderer::Texture::Desc { .format = draw_image_format,
																				.usage = usage::COLOR_ATTACHMENT | usage::TRANSFER_SRC,
																				.extent = draw_image_extent } );
	auto draw_resolved_image_view = device.create_texture_view( draw_resolved_image, renderer::TextureView::Aspect::COLOR );

	BindlessManager bindless_manager( device, std::array { MAX_MESHES, MAX_MATERIALS } );

	const auto depth_resolved_image = bindless_manager.add_texture(
		device.create_texture( renderer::Texture::Desc { .format = renderer::Texture::Format::D32_SFLOAT,
														 .usage = usage::DEPTH_STENCIL_ATTACHMENT | usage::SAMPLED,
														 .extent = draw_image_extent } ) );

	const auto depth_pyramid = bindless_manager.add_texture(
		device.create_texture(
			renderer::Texture::Desc { .format = renderer::Texture::Format::R32_SFLOAT,
									  .usage = usage::SAMPLED | usage::STORAGE,
									  .extent = { .width = draw_image_extent.width / 2, .height = draw_image_extent.height / 2 },
									  .mips = 8 } ),
		true );

	index_buffer indices( device, 1024 * 1024 );

	renderer::PipelineManager pipelines( device, "../shaders/vulkan", bindless_manager );

	renderer::PipelineHandle mesh_pipeline;
	renderer::PipelineHandle mesh_pipeline_meshlet;
	renderer::PipelineHandle depth_pyramid_pipeline;
	renderer::PipelineHandle grid_pipeline;
	model model;
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
				mesh_pipeline_meshlet = pipelines.add( renderer::Pipeline::Desc { .color_format = draw_image_format,
																				  .depth_format = depth_image.get_format(),
																				  .push_constants_size = sizeof( push_constants ) },
													   { { "mesh.task", renderer::ShaderStage::TASK },
														 { "mesh.mesh", renderer::ShaderStage::MESH },
														 { "mesh.frag", renderer::ShaderStage::FRAGMENT } } );
			},
			[ & ]
			{
				depth_pyramid_pipeline = pipelines.add(
					renderer::Pipeline::Desc { .push_constants_size = sizeof( depth_pyramid_constants ) },
					{ { "depth_pyramid.comp", renderer::ShaderStage::COMPUTE } } );
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
			[ & ] { model = load_model( device, bindless_manager, indices ); } );
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

		const bool mesh_shading = ui_state.mesh_shading;

		if ( !mesh_shading )
		{
			for ( auto& instances : instances_per_lod )
			{
				instances.clear();
			}

			for ( int z = 0; z < ui_state.z_miku; ++z )
			{
				for ( int x = 0; x < ui_state.x_miku; ++x )
				{
					const auto pos = glm::vec3( x - ( ui_state.x_miku / 2 ), 0, z - ( ui_state.z_miku / 2 ) );
					const auto visible = std::all_of(
						begin( constants.frustum ),
						end( constants.frustum ),
						[ radius = model.bounding_sphere.w, center = glm::vec3( model.bounding_sphere ) + pos ]( const glm::vec4& plane )
						{ return glm::dot( plane, glm::vec4( center, 1.f ) ) > -radius; } );

					if ( visible )
					{
						const auto distance = glm::distance( camera.position, pos );
						int lod = instances_per_lod.size() - 1;
						while ( distance < ui_state.lod_cutoff * lod )
						{
							--lod;
						}
						instances_per_lod[ lod ].push_back( glm::translate( glm::identity<glm::mat4>(), pos ) );
					}
				}
			}
		}

		auto [ image_index, swapchain_image, swapchain_image_view ] = swapchain.acquire();

		const uint32_t query_pool_offset = image_index * Timestamps::COUNT;
		std::array<uint64_t, Timestamps::COUNT> timestamps;
		if ( frame_count > renderer::MAX_FRAMES_IN_FLIGHT )
		{
			device.get_query_pool_results( query_pools[ image_index ], 0, timestamps );
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

		draw_ui( ui_state, props, bindless_manager, indices, model.meshes, timestamps, timestamp_period );

		auto command_buffer = device.grab_command_buffer();
		command_buffer->reset();
		command_buffer->begin();

		command_buffer->reset_query_pool( query_pools[ image_index ], 0, Timestamps::COUNT );
		command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::FRAME_START );

		command_buffer->transition_texture( draw_image,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( draw_resolved_image,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( depth_image,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::DEPTH_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( depth_resolved_image.texture,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::DEPTH_ATTACHMENT_OPTIMAL );

		command_buffer->begin_rendering(
			draw_image_extent,
			renderer::RenderAttachment { .target = draw_image_view,
										 .resolve_target = draw_resolved_image_view,
										 .clear_value = { { 0.1f, 0.1f, 0.1f, 1.f } } },
			renderer::RenderAttachment { .target = depth_image_view, .resolve_target = depth_resolved_image.handles.view } );

		command_buffer->set_viewport( draw_image_extent );
		command_buffer->set_scissor( draw_image_extent );

		int mesh_draws = 0;

		if ( mesh_shading )
		{
			command_buffer->bind_pipeline( pipelines.get( mesh_pipeline_meshlet ), bindless_manager );
			for ( int i = 0; i < model.meshes.size(); ++i )
			{
				OPTICK_GPU_EVENT( "Draw Meshlet" );
				const push_constants constants { .mesh = model.meshes[ i ].handle,
												 .material = model.materials[ i ].handle,
												 .scene_constants = constants_address,
												 .instance_buffer = instances_address,
												 .lod_cutoff = ui_state.lod_cutoff };
				command_buffer->push_constants( pipelines.get( mesh_pipeline_meshlet ), constants );
				command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_MESHES + mesh_draws );
				command_buffer->draw_mesh_task( ui_state.x_miku, 1, ui_state.z_miku );
				++mesh_draws;
				for ( int lod = 1; lod < 4; ++lod )
				{
					command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_MESHES + mesh_draws );
					++mesh_draws;
				}
			}
		}
		else
		{
			command_buffer->bind_pipeline( pipelines.get( mesh_pipeline ), bindless_manager );
			command_buffer->bind_index_buffer( indices.buffer );
			for ( int i = 0; i < model.meshes.size(); ++i )
			{
				OPTICK_GPU_EVENT( "Draw Mesh" );
				const push_constants constants { .mesh = model.meshes[ i ].handle,
												 .material = model.materials[ i ].handle,
												 .scene_constants = constants_address,
												 .instance_buffer = instances_address };

				command_buffer->push_constants( pipelines.get( mesh_pipeline ), constants );
				uint32_t base_instance = 0;
				for ( int lod = 0; lod < instances_per_lod.size(); ++lod )
				{
					command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_MESHES + mesh_draws );
					if ( !instances_per_lod[ lod ].empty() )
					{
						command_buffer->draw_indexed( model.meshes[ i ].gpu_data.lods[ lod ].indices_count,
													  instances_per_lod[ lod ].size(),
													  model.meshes[ i ].gpu_data.lods[ lod ].indices_offset,
													  base_instance );
						base_instance += instances_per_lod[ lod ].size();
					}
					++mesh_draws;
				}
			}
		}
		command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_MESHES + mesh_draws );

		{
			OPTICK_GPU_EVENT( "Draw Grid" );
			command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_GRID );
			command_buffer->bind_pipeline( pipelines.get( grid_pipeline ), bindless_manager );
			command_buffer->push_constants( pipelines.get( grid_pipeline ), grid_constants { .scene_constants = constants_address } );
			command_buffer->draw( 4 );
		}

		command_buffer->end_rendering();
		command_buffer->transition_texture( depth_resolved_image.texture,
											renderer::Texture::Layout::DEPTH_ATTACHMENT_OPTIMAL,
											renderer::Texture::Layout::SHADER_READ_ONLY_OPTIMAL );

		{
			command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DEPTH_PYRAMID );
			OPTICK_GPU_EVENT( "Depth Pyramid" );
			command_buffer->bind_pipeline( pipelines.get( depth_pyramid_pipeline ), bindless_manager );
			auto extent = depth_resolved_image.texture.get_extent();
			for ( int mip = 0; mip < depth_pyramid.mips.size(); ++mip )
			{
				command_buffer->transition_texture( depth_pyramid.texture,
													renderer::Texture::Layout::UNDEFINED,
													renderer::Texture::Layout::GENERAL,
													mip );
				extent.width /= 2;
				extent.height /= 2;
				const depth_pyramid_constants constants { .depth_in = mip == 0 ? depth_resolved_image.handles.texture_index
																			   : depth_pyramid.mips[ mip - 1 ].texture_index,
														  .depth_out = depth_pyramid.mips[ mip ].storage_index,
														  .size = glm::aligned_vec2( extent.width, extent.height ) };
				command_buffer->push_constants( pipelines.get( depth_pyramid_pipeline ), constants );
				command_buffer->dispatch( calc_group_count( extent.width ), calc_group_count( extent.height ), 1 );

				command_buffer->transition_texture( depth_pyramid.texture,
													renderer::Texture::Layout::GENERAL,
													renderer::Texture::Layout::SHADER_READ_ONLY_OPTIMAL,
													mip );
			}
		}

		command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::DRAW_UI );
		command_buffer->begin_rendering( draw_image_extent, renderer::RenderAttachment { .target = draw_resolved_image_view } );

		{
			OPTICK_GPU_EVENT( "Draw UI" );
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData( ImGui::GetDrawData(), command_buffer->get_impl() );
		}
		command_buffer->end_rendering();

		command_buffer->write_timestamp( query_pools[ image_index ], Timestamps::BLIT );

		command_buffer->transition_texture( draw_resolved_image,
											renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL,
											renderer::Texture::Layout::TRANSFER_SRC_OPTIMAL );
		command_buffer->transition_texture( swapchain_image,
											renderer::Texture::Layout::UNDEFINED,
											renderer::Texture::Layout::TRANSFER_DST_OPTIMAL );

		command_buffer->blit_texture( draw_resolved_image, swapchain_image );

		command_buffer->transition_texture( swapchain_image,
											renderer::Texture::Layout::TRANSFER_DST_OPTIMAL,
											renderer::Texture::Layout::PRESENT_SRC );

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
