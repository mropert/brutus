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
#include <memory>
#include <numbers>
#include <optick.h>
#include <print>
#include <span>
#include <stdexcept>

//

#include <tbb/tbb.h>

//

#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/command_buffer.h>
#include <renderer/device.h>
#include <renderer/sampler.h>
#include <renderer/shader.h>
#include <renderer/swapchain.h>
#include <renderer/texture.h>

renderer::raii::ShaderCode
compile_shader( const renderer::ShaderCompiler& compiler, const std::filesystem::path& path, renderer::ShaderStage stage )
{
	OPTICK_EVENT();
	std::string source;
	{
		OPTICK_EVENT( "read_shader" );
		std::ifstream istream( path );
		if ( !istream )
		{
			throw renderer::Error( "Couldn't open shader file: " + path.string() );
		}
		source.assign( ( std::istreambuf_iterator<char>( istream ) ), std::istreambuf_iterator<char>() );
	}
	{
		OPTICK_EVENT( "compile_shader" );
		auto compilation_result = compiler.compile( stage, source, path.string() );
		if ( !compilation_result )
		{
			throw renderer::Error( "Shader compilation failed: " + compilation_result.error() );
		}
		return std::move( compilation_result.value() );
	}
}

struct gpu_vertex_constants
{
	glm::mat4 worldMatrix;
	VkDeviceAddress vertexBuffer;
	std::byte padding[ 8 ];
};

static_assert( sizeof( gpu_vertex_constants ) % 16 == 0, "GPU vertex constants should be a multiple of 16 bytes" );

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
	gpu_vertex_constants vertex_constants;
	glm::vec3 camera_position;
	float padding;
	glm::vec4 main_light;
	gpu_material material;
};

renderer::raii::Pipeline
create_pipeline( renderer::Device& device, renderer::BindlessManager& tex_manager, renderer::Texture::Format draw_image_format )
{
	OPTICK_EVENT( "create_pipeline" );
	renderer::ShaderCompiler compiler;

	renderer::raii::ShaderCode vertex_shader;
	renderer::raii::ShaderCode fragment_shader;
	tbb::parallel_invoke( [ & ] { vertex_shader = compile_shader( compiler, "../shaders/vulkan/mesh.vs", renderer::ShaderStage::VERTEX ); },
						  [ & ]
						  {
							  fragment_shader = compile_shader( compiler, "../shaders/vulkan/mesh.fs", renderer::ShaderStage::FRAGMENT );
							  ;
						  } );

	return device.create_pipeline( renderer::Pipeline::Desc { .color_format = draw_image_format,
															  .depth_format = renderer::Texture::Format::D32_SFLOAT,
															  .push_constants_size = sizeof( push_constants ) },
								   vertex_shader,
								   fragment_shader,
								   tex_manager );
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

	void upload_vertices()
	{
		OPTICK_EVENT();

		std::size_t alloc_size = 0;
		for ( auto& mesh : _meshes )
		{
			alloc_size += sizeof( mesh_vertex ) * mesh.positions.length;
			alloc_size += sizeof( uint32_t ) * mesh.indices.length;
		}

		_vertex_cmd_buffer->reset();
		_vertex_cmd_buffer->begin();

		using BufferUsage = renderer::Buffer::Usage;
		_vertex_upload_buffer = _device.create_buffer( BufferUsage::TRANSFER_SRC, alloc_size, true );
		const auto ptr = static_cast<std::byte*>( _vertex_upload_buffer.get_mapped_address() );
		const auto& gltf_buffer = std::get<fastgltf::sources::Array>( _asset.buffers[ 0 ].data ).bytes;

		std::size_t offset = 0;
		for ( int i = 0; i < _meshes.size(); ++i )
		{
			const auto& mesh = _meshes[ i ];
			auto& gpu_data = _gpu_meshes[ i ];
			gpu_data.vertices = _device.create_buffer( BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST
														   | BufferUsage::SHADER_DEVICE_ADDRESS,
													   sizeof( mesh_vertex ) * mesh.positions.length );

			gpu_data.indices = _device.create_buffer( BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
													  sizeof( uint32_t ) * mesh.indices.length );

			{
				OPTICK_EVENT( "repack_vertices" );
				const auto verts = reinterpret_cast<mesh_vertex*>( ptr + offset );
				const auto positions = reinterpret_cast<const float*>( gltf_buffer.data() + mesh.positions.offset );
				const auto normals = reinterpret_cast<const float*>( gltf_buffer.data() + mesh.normals.offset );
				const auto tangents = reinterpret_cast<const float*>( gltf_buffer.data() + mesh.tangents.offset );
				const auto uvs = reinterpret_cast<const float*>( gltf_buffer.data() + mesh.uvs.offset );
				for ( int i = 0; i < mesh.positions.length; ++i )
				{
					verts[ i ].position[ 0 ] = positions[ i * 3 ];
					verts[ i ].position[ 1 ] = positions[ i * 3 + 1 ];
					verts[ i ].position[ 2 ] = positions[ i * 3 + 2 ];
					verts[ i ].normal[ 0 ] = normals[ i * 3 ];
					verts[ i ].normal[ 1 ] = normals[ i * 3 + 1 ];
					verts[ i ].normal[ 2 ] = normals[ i * 3 + 2 ];
					verts[ i ].u = uvs[ i * 2 ];
					verts[ i ].v = uvs[ i * 2 + 1 ];
					verts[ i ].tangent[ 0 ] = tangents[ i * 4 ];
					verts[ i ].tangent[ 1 ] = tangents[ i * 4 + 1 ];
					verts[ i ].tangent[ 2 ] = tangents[ i * 4 + 2 ];
					verts[ i ].tangent[ 3 ] = tangents[ i * 4 + 3 ];
				}
			}
			_vertex_cmd_buffer->copy_buffer( _vertex_upload_buffer,
											 offset,
											 sizeof( mesh_vertex ) * mesh.positions.length,
											 gpu_data.vertices );
			offset += sizeof( mesh_vertex ) * mesh.positions.length;

			memcpy( ptr + offset, gltf_buffer.data() + mesh.indices.offset, sizeof( uint32_t ) * mesh.indices.length );
			_vertex_cmd_buffer->copy_buffer( _vertex_upload_buffer, offset, sizeof( uint32_t ) * mesh.indices.length, gpu_data.indices );
			offset += sizeof( uint32_t ) * mesh.indices.length;
		}

		_vertex_cmd_buffer->end();
		_device.submit( *_vertex_cmd_buffer, _vertex_upload_fence );
	}

	void decompress_textures()
	{
		OPTICK_EVENT();

		const auto& gltf_buffer = std::get<fastgltf::sources::Array>( _asset.buffers[ 0 ].data ).bytes;
		tbb::parallel_for(
			size_t( 0 ),
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

	glm::mat4 getViewMatrix()
	{
		glm::mat4 cameraTranslation = glm::translate( glm::mat4( 1.f ), position );
		glm::mat4 cameraRotation = getRotationMatrix();
		return glm::inverse( cameraTranslation * cameraRotation );
	}
	glm::mat4 getRotationMatrix()
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
			OPTICK_START_THREAD( "TBB Worker" );
		}
	}
	void on_scheduler_exit( bool is_worker ) override
	{
		if ( is_worker )
		{
			OPTICK_STOP_THREAD();
		}
	}
};

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
	OPTICK_POP();

	OPTICK_PUSH( "create_swapchain" );
	constexpr auto swapchain_image_format = renderer::Texture::Format::R8G8B8A8_SRGB;
	renderer::Swapchain swapchain( device, swapchain_image_format );
	OPTICK_POP();

	const auto draw_image_extent = device.get_extent();
	constexpr auto draw_image_format = renderer::Texture::Format::R16G16B16A16_SFLOAT;
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

	renderer::raii::Pipeline pipeline;
	std::vector<gpu_mesh_data> meshes;
	tbb::parallel_invoke( [ & ] { pipeline = create_pipeline( device, tex_manager, draw_image_format ); },
						  [ & ] { meshes = load_model( device, tex_manager ); } );

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
			else
			{
				camera.processSDLEvent( event );
			}
		}
		const auto old_time = time;
		time = std::chrono::steady_clock::now();
		using namespace std::chrono_literals;
		const auto delta_time = duration_cast<std::chrono::microseconds>( time - old_time ).count()
			/ static_cast<float>( duration_cast<std::chrono::microseconds>( 1s ).count() );
		time_of_day += delta_time;
		if ( time_of_day > day_period )
		{
			time_of_day -= day_period;
		}
		camera.update( delta_time );

		auto [ image_index, swapchain_image, swapchain_image_view ] = swapchain.acquire();
		auto command_buffer = device.grab_command_buffer();

		command_buffer->reset();
		command_buffer->begin();

		command_buffer->transition_texture( draw_image, renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( draw_resolved_image, renderer::Texture::Layout::COLOR_ATTACHMENT_OPTIMAL );
		command_buffer->transition_texture( depth_image, renderer::Texture::Layout::DEPTH_ATTACHMENT_OPTIMAL );

		command_buffer->begin_rendering( draw_image_extent,
										 renderer::RenderAttachment { .target = draw_image_view,
																	  .resolve_target = draw_resolved_image_view,
																	  .clear_value = { { 0.1f, 0.1f, 0.1f, 1.f } } },
										 renderer::RenderAttachment { .target = depth_image_view } );

		command_buffer->bind_pipeline( pipeline, tex_manager );
		command_buffer->set_viewport( draw_image_extent );
		command_buffer->set_scissor( draw_image_extent );

		const glm::mat4 view = camera.getViewMatrix();
		glm::mat4 projection = glm::perspective( glm::radians( 70.f ),
												 static_cast<float>( draw_image_extent.width ) / draw_image_extent.height,
												 10000.f,
												 0.1f );

		// invert the Y direction on projection matrix so that we are more similar
		// to opengl and gltf axis
		projection[ 1 ][ 1 ] *= -1;

		const float light_time = time_of_day / day_period;
		const glm::vec4 main_light( 20.f * std::cosf( light_time * std::numbers::pi * 2.f ),
									2.f,
									10.f * std::sinf( light_time * std::numbers::pi * 2.f ),
									2.5f );

		for ( const auto& mesh : meshes )
		{
			OPTICK_GPU_EVENT( "Draw Mesh" );

			const push_constants constants { .vertex_constants { .worldMatrix = projection * view,
																 .vertexBuffer = mesh.vertices.get_device_address() },
											 .camera_position = camera.position,
											 .main_light = main_light,
											 .material = mesh.material };

			command_buffer->push_constants( pipeline, constants );
			command_buffer->draw_indexed( mesh.indices );
		}

		command_buffer->end_rendering();

		command_buffer->transition_texture( draw_resolved_image, renderer::Texture::Layout::TRANSFER_SRC_OPTIMAL );
		command_buffer->transition_texture( swapchain_image, renderer::Texture::Layout::TRANSFER_DST_OPTIMAL );

		command_buffer->blit_texture( draw_resolved_image, swapchain_image );

		command_buffer->transition_texture( swapchain_image, renderer::Texture::Layout::PRESENT_SRC );

		command_buffer->end();

		swapchain.submit( *command_buffer );

		swapchain.present();

		++frame_count;
	}

	// Wait for the GPU to finish before we roll all the destructors
	device.wait_idle();

	OPTICK_SHUTDOWN();

	return 0;
}
