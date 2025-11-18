#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3_image/SDL_image.h>
#include <array>
#include <expected>
#include <fastgltf/core.hpp>
#include <filesystem>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <optick.h>
#include <print>
#include <shaderc/shaderc.hpp>
#include <span>
#include <stdexcept>

//

#include <renderer/command_buffer.h>
#include <renderer/device.h>
#include <renderer/swapchain.h>
#include <renderer/texture.h>

namespace raii
{
	struct VkObjectDeleter
	{
		::VkDevice device;
		void operator()( ::VkCommandPool command_pool ) const { vkDestroyCommandPool( device, command_pool, nullptr ); }
		void operator()( ::VkDescriptorPool pool ) const { vkDestroyDescriptorPool( device, pool, nullptr ); }
		void operator()( ::VkDescriptorSetLayout layout ) const { vkDestroyDescriptorSetLayout( device, layout, nullptr ); }
		void operator()( ::VkFence fence ) const { vkDestroyFence( device, fence, nullptr ); }
		void operator()( ::VkImageView image_view ) const { vkDestroyImageView( device, image_view, nullptr ); }
		void operator()( ::VkPipeline pipeline ) const { vkDestroyPipeline( device, pipeline, nullptr ); }
		void operator()( ::VkPipelineLayout layout ) const { vkDestroyPipelineLayout( device, layout, nullptr ); }
		void operator()( ::VkSampler sampler ) const { vkDestroySampler( device, sampler, nullptr ); }
		void operator()( ::VkSemaphore semaphore ) const { vkDestroySemaphore( device, semaphore, nullptr ); }
		void operator()( ::VkShaderModule shader ) const { vkDestroyShaderModule( device, shader, nullptr ); }
	};
	template <typename T>
	using VkObject = std::unique_ptr<std::remove_pointer_t<T>, VkObjectDeleter>;

	template <typename T>
	auto make( ::VkDevice device, T object )
	{
		return VkObject<T> { object, VkObjectDeleter( device ) };
	}

	using VkCommandPool = VkObject<::VkCommandPool>;
	using VkDescriptorPool = VkObject<::VkDescriptorPool>;
	using VkDescriptorSetLayout = VkObject<::VkDescriptorSetLayout>;
	using VkFence = VkObject<::VkFence>;
	using VkImageView = VkObject<::VkImageView>;
	using VkPipeline = VkObject<::VkPipeline>;
	using VkPipelineLayout = VkObject<::VkPipelineLayout>;
	using VkSampler = VkObject<::VkSampler>;
	using VkSemaphore = VkObject<::VkSemaphore>;
	using VkShaderModule = VkObject<::VkShaderModule>;

	struct VulkanImage
	{
		vma::raii::Image image;
		VkImageView view;
		::VkExtent3D extent;
		::VkFormat format;
	};
}

std::expected<raii::VkFence, VkResult> make_fence( VkDevice device, VkFenceCreateFlags flags )
{
	const VkFenceCreateInfo fenceCreateInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
	VkFence fence;
	const auto ret = vkCreateFence( device, &fenceCreateInfo, nullptr, &fence );
	if ( ret )
	{
		return std::unexpected( ret );
	}
	return raii::make( device, fence );
}

std::expected<raii::VkSemaphore, VkResult> make_semaphore( VkDevice device )
{
	const VkSemaphoreCreateInfo semaphoreCreateInfo = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	VkSemaphore semaphore;
	const auto ret = vkCreateSemaphore( device, &semaphoreCreateInfo, nullptr, &semaphore );
	if ( ret )
	{
		return std::unexpected( ret );
	}
	return raii::make( device, semaphore );
}

std::expected<vma::raii::Buffer, VkResult>
make_buffer( VmaAllocator allocator, size_t size, VkBufferUsageFlags buffer_usage, VmaMemoryUsage memory_usage )
{
	const VkBufferCreateInfo bufferInfo { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = buffer_usage };
	const VmaAllocationCreateInfo vmaallocInfo = { .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT, .usage = memory_usage };

	VkBuffer buffer {};
	VmaAllocation allocation {};
	VmaAllocationInfo info {};
	const auto ret = vmaCreateBuffer( allocator, &bufferInfo, &vmaallocInfo, &buffer, &allocation, &info );
	if ( ret )
	{
		return std::unexpected( ret );
	}

	return vma::raii::make( allocator, buffer, allocation, info );
}

std::expected<raii::VulkanImage, VkResult> make_image( VkDevice device,
													   VmaAllocator allocator,
													   VkFormat format,
													   VkExtent3D extent,
													   VkSampleCountFlagBits samples,
													   VkImageUsageFlags image_usage,
													   VmaMemoryUsage memory_usage,
													   VkImageAspectFlags aspect_flags )
{
	const VkImageCreateInfo info { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
								   .imageType = VK_IMAGE_TYPE_2D,
								   .format = format,
								   .extent = extent,
								   .mipLevels = 1,
								   .arrayLayers = 1,
								   .samples = samples,
								   .tiling = VK_IMAGE_TILING_OPTIMAL,
								   .usage = image_usage };

	const VmaAllocationCreateInfo create_info { .usage = memory_usage, .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT };

	VkImage image {};
	VmaAllocation allocation {};
	VmaAllocationInfo allocation_info {};
	auto ret = vmaCreateImage( allocator, &info, &create_info, &image, &allocation, &allocation_info );
	if ( ret )
	{
		return std::unexpected( ret );
	}

	raii::VulkanImage allocated_image;
	allocated_image.image = vma::raii::make( allocator, image, allocation, allocation_info );
	allocated_image.extent = extent;
	allocated_image.format = format;

	const VkImageViewCreateInfo image_view_info {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = { .aspectMask = aspect_flags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 }
	};

	VkImageView view;
	ret = vkCreateImageView( device, &image_view_info, nullptr, &view );
	if ( ret )
	{
		return std::unexpected( ret );
	}
	allocated_image.view = raii::make( device, view );
	return allocated_image;
}

std::expected<raii::VkSampler, VkResult> make_sampler( VkDevice device, VkFilter filter )
{
	const VkSamplerCreateInfo info { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = filter, .minFilter = filter };
	VkSampler sampler = nullptr;
	const auto res = vkCreateSampler( device, &info, nullptr, &sampler );
	if ( res )
	{
		return std::unexpected( res );
	}
	return raii::make( device, sampler );
}

std::expected<raii::VkShaderModule, VkResult>
make_shader( VkDevice device, const std::string_view source, VkShaderStageFlagBits type, const char* filename )
{
	OPTICK_EVENT();
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;

	options.SetOptimizationLevel( shaderc_optimization_level_performance );

	const auto result = compiler.CompileGlslToSpv( source.data(),
												   source.size(),
												   type == VK_SHADER_STAGE_VERTEX_BIT ? shaderc_glsl_vertex_shader
																					  : shaderc_glsl_fragment_shader,
												   filename,
												   options );

	if ( result.GetCompilationStatus() != shaderc_compilation_status_success )
	{
		std::println( stderr, "Shader compilation failed: {}", result.GetErrorMessage() );
		return {};
	}

	VkShaderModuleCreateInfo shader_info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
											 .codeSize = std::distance( result.begin(), result.end() ) * sizeof( uint32_t ),
											 .pCode = result.begin() };

	VkShaderModule shader {};
	const auto ret = vkCreateShaderModule( device, &shader_info, nullptr, &shader );
	if ( ret )
	{
		return std::unexpected( ret );
	}
	return raii::make( device, shader );
}

std::expected<raii::VkShaderModule, VkResult> make_shader( VkDevice device, const std::filesystem::path& path, VkShaderStageFlagBits type )
{
	OPTICK_EVENT();
	std::string source;
	{
		std::ifstream istream( path );
		if ( !istream )
		{
			// TODO: return proper file error
			return std::unexpected( VK_ERROR_UNKNOWN );
		}
		source.assign( ( std::istreambuf_iterator<char>( istream ) ), std::istreambuf_iterator<char>() );
	}
	return make_shader( device, std::string_view( source ), type, path.filename().string().c_str() );
}

std::expected<raii::VkPipeline, VkResult> make_pipeline( VkDevice device,
														 VkFormat color_format,
														 VkPipelineLayout layout,
														 VkShaderModule vertex_shader,
														 VkShaderModule fragment_shader )
{
	OPTICK_EVENT();
	const VkPipelineVertexInputStateCreateInfo vertex_input { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
	const VkPipelineInputAssemblyStateCreateInfo ia { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
													  .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
	const VkPipelineViewportStateCreateInfo viewport { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
													   .viewportCount = 1,
													   .scissorCount = 1 };
	const VkPipelineRasterizationStateCreateInfo rasterizer { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
															  .polygonMode = VK_POLYGON_MODE_FILL,
															  .cullMode = VK_CULL_MODE_NONE,
															  .frontFace = VK_FRONT_FACE_CLOCKWISE,
															  .lineWidth = 1.f };
	const VkPipelineMultisampleStateCreateInfo multisampling { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
															   .rasterizationSamples = VK_SAMPLE_COUNT_4_BIT,
															   .minSampleShading = 1.0f };
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
																  .depthTestEnable = VK_TRUE,
																  .depthWriteEnable = VK_TRUE,
																  .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL,
																  .maxDepthBounds = 1.f };

	const VkPipelineColorBlendAttachmentState blend_attachment { .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
																	 | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
	const VkPipelineColorBlendStateCreateInfo blend_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
															  .logicOp = VK_LOGIC_OP_COPY,
															  .attachmentCount = 1,
															  .pAttachments = &blend_attachment };
	const VkPipelineRenderingCreateInfo render_info { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
													  .colorAttachmentCount = 1,
													  .pColorAttachmentFormats = &color_format,
													  .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT };

	const std::array<VkDynamicState, 2> state { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	const VkPipelineDynamicStateCreateInfo dynamic_state { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
														   .dynamicStateCount = state.size(),
														   .pDynamicStates = state.data() };

	const std::array<VkPipelineShaderStageCreateInfo, 2> shaders { { { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
																	   .stage = VK_SHADER_STAGE_VERTEX_BIT,
																	   .module = vertex_shader,
																	   .pName = "main" },
																	 { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
																	   .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
																	   .module = fragment_shader,
																	   .pName = "main" } } };

	const VkGraphicsPipelineCreateInfo pipelineInfo = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
														.pNext = &render_info,
														.stageCount = shaders.size(),
														.pStages = shaders.data(),
														.pVertexInputState = &vertex_input,
														.pInputAssemblyState = &ia,
														.pViewportState = &viewport,
														.pRasterizationState = &rasterizer,
														.pMultisampleState = &multisampling,
														.pDepthStencilState = &depth_stencil,
														.pColorBlendState = &blend_state,
														.pDynamicState = &dynamic_state,
														.layout = layout };

	VkPipeline gfx_pipeline {};
	const auto ret = vkCreateGraphicsPipelines( device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &gfx_pipeline );
	if ( ret )
	{
		return std::unexpected( ret );
	}
	return raii::make( device, gfx_pipeline );
}

struct TextureManager
{
	explicit TextureManager( VkDevice device )
		: m_device( device )
	{
		const std::array<VkDescriptorSetLayoutBinding, 2> bindings { { { .binding = 0,
																		 .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
																		 .descriptorCount = max_textures,
																		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
																	   { .binding = 1,
																		 .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
																		 .descriptorCount = 1,
																		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT } } };

		const VkDescriptorSetLayoutCreateInfo layout_info { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
															.bindingCount = bindings.size(),
															.pBindings = bindings.data() };

		VkDescriptorSetLayout set = nullptr;
		auto res = vkCreateDescriptorSetLayout( m_device, &layout_info, nullptr, &set );
		if ( res )
		{
			throw std::runtime_error( "Failed to create descriptor set layout: " + std::to_string( res ) );
		}
		m_layout = raii::make( device, set );

		const std::array<VkDescriptorPoolSize, 2> pools { { { .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = max_textures },
															{ .type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = 1 } } };
		const VkDescriptorPoolCreateInfo pool_info { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
													 .maxSets = max_textures + 1,
													 .poolSizeCount = pools.size(),
													 .pPoolSizes = pools.data() };
		VkDescriptorPool pool = nullptr;
		res = vkCreateDescriptorPool( m_device, &pool_info, nullptr, &pool );
		if ( res )
		{
			throw std::runtime_error( "Failed to create descriptor pool: " + std::to_string( res ) );
		}
		m_pool = raii::make( device, pool );

		const VkDescriptorSetAllocateInfo alloc_info { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
													   .descriptorPool = m_pool.get(),
													   .descriptorSetCount = 1,
													   .pSetLayouts = &set };
		res = vkAllocateDescriptorSets( m_device, &alloc_info, &m_desc );
		if ( res )
		{
			throw std::runtime_error( "Failed to allocate descriptor set: " + std::to_string( res ) );
		}

		m_linear_sampler = *make_sampler( m_device, VK_FILTER_LINEAR );
	}

	uint32_t add_texture( raii::VulkanImage&& tex )
	{
		m_textures.push_back( std::move( tex ) );
		return m_textures.size() - 1;
	}

	void update_descriptors()
	{
		std::vector<VkDescriptorImageInfo> images_info;
		images_info.reserve( m_textures.size() );
		for ( const auto& tex : m_textures )
		{
			images_info.emplace_back( nullptr, tex.view.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
		}

		const VkDescriptorImageInfo sampler_info { .sampler = m_linear_sampler.get() };

		const std::array<VkWriteDescriptorSet, 2> writes { { { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
															   .dstSet = m_desc,
															   .dstBinding = 0,
															   .descriptorCount = static_cast<uint32_t>( images_info.size() ),
															   .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
															   .pImageInfo = images_info.data() },
															 { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
															   .dstSet = m_desc,
															   .dstBinding = 1,
															   .descriptorCount = 1,
															   .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
															   .pImageInfo = &sampler_info } } };

		vkUpdateDescriptorSets( m_device, writes.size(), writes.data(), 0, nullptr );
	}

	static inline constexpr uint32_t max_textures = 8192;

	VkDevice m_device;
	raii::VkDescriptorSetLayout m_layout;
	raii::VkDescriptorPool m_pool;
	VkDescriptorSet m_desc = nullptr;
	std::vector<raii::VulkanImage> m_textures;
	raii::VkSampler m_linear_sampler;
};

void transition_image( VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout )
{
	const VkImageAspectFlags aspectMask = ( newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL ) ? VK_IMAGE_ASPECT_DEPTH_BIT
																									: VK_IMAGE_ASPECT_COLOR_BIT;
	const VkImageMemoryBarrier2 imageBarrier { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
											   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
											   .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
											   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
											   .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
											   .oldLayout = currentLayout,
											   .newLayout = newLayout,
											   .image = image,
											   .subresourceRange = {
												   .aspectMask = aspectMask,
												   .levelCount = VK_REMAINING_MIP_LEVELS,
												   .layerCount = VK_REMAINING_ARRAY_LAYERS,
											   } };

	const VkDependencyInfo depInfo { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
									 .imageMemoryBarrierCount = 1,
									 .pImageMemoryBarriers = &imageBarrier };

	vkCmdPipelineBarrier2( cmd, &depInfo );
}

void blit_image( VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D src_size, VkExtent2D dst_size )
{
	const VkImageBlit2 blitRegion {
		.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
		.srcSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.srcOffsets = { {}, { static_cast<int32_t>( src_size.width ), static_cast<int32_t>( src_size.height ), 1 } },
		.dstSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.dstOffsets = { {}, { static_cast<int32_t>( dst_size.width ), static_cast<int32_t>( dst_size.height ), 1 } },
	};

	const VkBlitImageInfo2 blitInfo { .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
									  .srcImage = source,
									  .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
									  .dstImage = destination,
									  .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
									  .regionCount = 1,
									  .pRegions = &blitRegion,
									  .filter = VK_FILTER_LINEAR };

	vkCmdBlitImage2( cmd, &blitInfo );
}

void resolve_image( VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D size )
{
	OPTICK_GPU_EVENT( "Resolve Image" );
	const VkImageResolve resolveRegion {
		.srcSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.dstSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.extent = { .width = size.width, .height = size.height, .depth = 1 }
	};

	vkCmdResolveImage( cmd,
					   source,
					   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					   destination,
					   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					   1,
					   &resolveRegion );
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

struct pbr_data
{
	glm::vec4 base_color;
	float metalness;
	float roughness;
};

struct mesh
{
	mesh_attribute positions;
	mesh_attribute normals;
	mesh_attribute tangents;
	mesh_attribute uvs;
	mesh_attribute indices;
	mesh_texture albedo;
	mesh_texture normal;
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

struct gpu_material
{
	pbr_data pbr_data;
	uint32_t albedo_texture_index;
	uint32_t normal_texture_index;
};

struct gpu_mesh_data
{
	vma::raii::Buffer vertices;
	VkDeviceAddress vertices_address;
	vma::raii::Buffer indices;
	uint32_t indices_count;
	gpu_material material;
};

std::expected<raii::VulkanImage, VkResult>
make_image_from_surface( VkDevice device, VmaAllocator allocator, const SDL_Surface& surface, bool sRGB )
{
	OPTICK_EVENT();
	const auto format = sdl_to_vulkan_format( sdl_gpu_format( surface.format ), sRGB );
	if ( format == VK_FORMAT_UNDEFINED )
	{
		return std::unexpected( VK_ERROR_FORMAT_NOT_SUPPORTED );
	}
	return make_image( device,
					   allocator,
					   format,
					   VkExtent3D( surface.w, surface.h, 1 ),
					   VK_SAMPLE_COUNT_1_BIT,
					   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					   VMA_MEMORY_USAGE_GPU_ONLY,
					   VK_IMAGE_ASPECT_COLOR_BIT );
}

std::size_t upload_image_from_surface( VkCommandBuffer cmd,
									   VkBuffer transfer_buffer,
									   std::byte* buffer,
									   std::size_t offset,
									   VkImage image,
									   const SDL_Surface& surface )
{
	OPTICK_EVENT();
	const auto gpu_format = sdl_gpu_format( surface.format );
	const auto pitch = surface.w * get_format_size( sdl_to_vulkan_format( gpu_format ) );
	SDL_ConvertPixels( surface.w, surface.h, surface.format, surface.pixels, surface.pitch, gpu_format, buffer + offset, pitch );
	transition_image( cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
	const VkBufferImageCopy copy { .bufferOffset = offset,
								   .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
								   .imageExtent = VkExtent3D( surface.w, surface.h, 1 ) };
	vkCmdCopyBufferToImage( cmd, transfer_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy );
	transition_image( cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

	return pitch * surface.h;
}

std::vector<gpu_mesh_data>
load_model( VkDevice device, VmaAllocator allocator, VkCommandBuffer cmd, VkFence fence, VkQueue queue, TextureManager& tex_manager )
{
	OPTICK_EVENT();
	const std::filesystem::path path = "../assets/player.glb";

	fastgltf::Parser parser;
	fastgltf::GltfDataBuffer data;
	{
		OPTICK_EVENT( "gltf_read_io" );
		auto res = fastgltf::GltfDataBuffer::FromPath( path );
		if ( res.error() != fastgltf::Error::None )
		{
			return {};
		}
		data = std::move( res.get() );
	}

	fastgltf::Asset asset;
	{
		OPTICK_EVENT( "gltf_parse" );
		auto res = parser.loadGltf( data, path.parent_path(), fastgltf::Options::None );
		if ( auto error = res.error(); error != fastgltf::Error::None )
		{
			return {};
		}
		asset = std::move( res.get() );
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

	std::vector<mesh> meshes;
	for ( const auto& mesh : asset.meshes )
	{
		auto& m = meshes.emplace_back();
		if ( mesh.primitives[ 0 ].materialIndex )
		{
			const auto& material = asset.materials[ *mesh.primitives[ 0 ].materialIndex ];
			if ( material.pbrData.baseColorTexture )
			{
				m.albedo = make_texture( asset, material.pbrData.baseColorTexture->textureIndex );
			}
			if ( material.normalTexture )
			{
				m.normal = make_texture( asset, material.normalTexture->textureIndex );
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
				m.positions = make_attribute( asset, attribute.accessorIndex );
			}
			else if ( attribute.name == "NORMAL" )
			{
				m.normals = make_attribute( asset, attribute.accessorIndex );
			}
			else if ( attribute.name == "TANGENT" )
			{
				m.tangents = make_attribute( asset, attribute.accessorIndex );
			}
			else if ( attribute.name == "TEXCOORD_0" )
			{
				m.uvs = make_attribute( asset, attribute.accessorIndex );
			}
		}
		if ( mesh.primitives[ 0 ].indicesAccessor )
		{
			m.indices = make_attribute( asset, *mesh.primitives[ 0 ].indicesAccessor );
		}
	}

	const auto& gltf_buffer = std::get<fastgltf::sources::Array>( asset.buffers[ 0 ].data ).bytes;

	std::size_t alloc_size = 0;
	{
		OPTICK_EVENT( "load_images" );
		for ( auto& mesh : meshes )
		{
			if ( const auto tex = IMG_Load_IO( SDL_IOFromConstMem( gltf_buffer.data() + mesh.albedo.offset, mesh.albedo.size ), true ) )
			{
				mesh.albedo.image.reset( tex );
				alloc_size += tex->w * tex->h * get_format_size( sdl_to_vulkan_format( sdl_gpu_format( tex->format ) ) );
			}
			if ( const auto tex = IMG_Load_IO( SDL_IOFromConstMem( gltf_buffer.data() + mesh.normal.offset, mesh.normal.size ), true ) )
			{
				mesh.normal.image.reset( tex );
				alloc_size += tex->w * tex->h * get_format_size( sdl_to_vulkan_format( sdl_gpu_format( tex->format ) ) );
			}
			alloc_size += sizeof( mesh_vertex ) * mesh.positions.length;
			alloc_size += sizeof( uint32_t ) * mesh.indices.length;
		}
	}

	vkResetFences( device, 1, &fence );
	vkResetCommandBuffer( cmd, 0 );
	const VkCommandBufferBeginInfo cmdBeginInfo { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
												  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	vkBeginCommandBuffer( cmd, &cmdBeginInfo );

	std::vector<gpu_mesh_data> gpu_meshes;
	const auto transfer_buffer = make_buffer( allocator, alloc_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY ).value();
	const auto ptr = static_cast<std::byte*>( transfer_buffer.get_deleter().info.pMappedData );
	std::size_t offset = 0;
	for ( const auto& mesh : meshes )
	{
		auto& gpu_data = gpu_meshes.emplace_back();
		gpu_data.vertices = make_buffer( allocator,
										 sizeof( mesh_vertex ) * mesh.positions.length,
										 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
											 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
										 VMA_MEMORY_USAGE_GPU_ONLY )
								.value();

		const VkBufferDeviceAddressInfo deviceAdressInfo { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
														   .buffer = VkBuffer( gpu_data.vertices.get() ) };
		gpu_data.vertices_address = vkGetBufferDeviceAddress( device, &deviceAdressInfo );

		gpu_data.indices = make_buffer( allocator,
										sizeof( uint32_t ) * mesh.indices.length,
										VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
										VMA_MEMORY_USAGE_GPU_ONLY )
							   .value();
		gpu_data.indices_count = mesh.indices.length;

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
		const VkBufferCopy vertexCopy { .srcOffset = offset, .dstOffset = 0, .size = sizeof( mesh_vertex ) * mesh.positions.length };
		vkCmdCopyBuffer( cmd, VkBuffer( transfer_buffer.get() ), VkBuffer( gpu_data.vertices.get() ), 1, &vertexCopy );
		offset += sizeof( mesh_vertex ) * mesh.positions.length;

		memcpy( ptr + offset, gltf_buffer.data() + mesh.indices.offset, sizeof( uint32_t ) * mesh.indices.length );
		const VkBufferCopy indexCopy { .srcOffset = offset, .dstOffset = 0, .size = sizeof( uint32_t ) * mesh.indices.length };
		vkCmdCopyBuffer( cmd, VkBuffer( transfer_buffer.get() ), VkBuffer( gpu_data.indices.get() ), 1, &indexCopy );
		offset += sizeof( uint32_t ) * mesh.indices.length;

		auto tex = *make_image_from_surface( device, allocator, *mesh.albedo.image, true );
		offset += upload_image_from_surface( cmd, transfer_buffer.get(), ptr, offset, tex.image.get(), *mesh.albedo.image );
		gpu_data.material.albedo_texture_index = tex_manager.add_texture( std::move( tex ) );

		tex = *make_image_from_surface( device, allocator, *mesh.normal.image, false );
		offset += upload_image_from_surface( cmd, transfer_buffer.get(), ptr, offset, tex.image.get(), *mesh.normal.image );
		gpu_data.material.normal_texture_index = tex_manager.add_texture( std::move( tex ) );

		gpu_data.material.pbr_data = mesh.pbr_data;
	}

	vkEndCommandBuffer( cmd );
	const VkCommandBufferSubmitInfo cmdSubmitInfo { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd };
	const VkSubmitInfo2 submitInfo {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.commandBufferInfoCount = 1,
		.pCommandBufferInfos = &cmdSubmitInfo,
	};

	vkQueueSubmit2( queue, 1, &submitInfo, fence );
	vkWaitForFences( device, 1, &fence, true, 9999999999 );

	return gpu_meshes;
}

struct gpu_vertex_constants
{
	glm::mat4 worldMatrix;
	VkDeviceAddress vertexBuffer;
	std::byte padding[ 8 ];
};

static_assert( sizeof( gpu_vertex_constants ) % 16 == 0, "GPU vertex constants should be a multiple of 16 bytes" );

int main()
{
	constexpr bool profile_startup = false;

	if ( profile_startup )
	{
		OPTICK_START_CAPTURE();
	}

	OPTICK_PUSH( "Startup" );

	renderer::Device device( "Brutus-NG [Vulkan]" );

	VkDevice vk_device = *device._device;
	VkPhysicalDevice physical_device = *device._physical_device;
	VkQueue gfx_queue = *device._gfx_queue;
	OPTICK_GPU_INIT_VULKAN( &vk_device, &physical_device, &gfx_queue, &device._gfx_queue_family_index, 1, nullptr );

	constexpr auto swapchain_image_format = renderer::Texture::Format::R8G8B8A8_SRGB;
	renderer::Swapchain swapchain( device, swapchain_image_format );
	VkSwapchainKHR vk_swapchain = *swapchain._swapchain;

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

	TextureManager tex_manager( vk_device );

	raii::VkPipelineLayout vk_pipeline_layout;
	raii::VkPipeline vk_pipeline;
	{
		OPTICK_EVENT( "create_vk_pipeline" );
		const std::array<VkPushConstantRange, 2> constants {
			{ { .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .size = sizeof( gpu_vertex_constants ) },
			  { .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = sizeof( gpu_vertex_constants ), .size = sizeof( gpu_material ) } }
		};
		VkPipelineLayout layout {};
		const auto desc_layout = tex_manager.m_layout.get();
		VkPipelineLayoutCreateInfo pipeline_layout_info { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
														  .setLayoutCount = 1,
														  .pSetLayouts = &desc_layout,
														  .pushConstantRangeCount = constants.size(),
														  .pPushConstantRanges = constants.data() };
		const auto ret = vkCreatePipelineLayout( vk_device, &pipeline_layout_info, nullptr, &layout );
		if ( ret )
		{
			std::println( stderr, "Cannot create pipeline layout: {}", static_cast<int>( ret ) );
			return 1;
		}
		vk_pipeline_layout = raii::make( vk_device, layout );

		const auto vertex_shader = make_shader( vk_device, "../shaders/vulkan/mesh.vs", VK_SHADER_STAGE_VERTEX_BIT ).value();
		const auto fragment_shader = make_shader( vk_device, "../shaders/vulkan/mesh.fs", VK_SHADER_STAGE_FRAGMENT_BIT ).value();

		vk_pipeline = make_pipeline( vk_device,
									 static_cast<VkFormat>( draw_image_format ),
									 layout,
									 vertex_shader.get(),
									 fragment_shader.get() )
						  .value();
	}


	auto transfer_command_buffer = device.grab_command_buffer();
	const auto transfer_fence = make_fence( vk_device, 0 ).value();

	const auto meshes = load_model( vk_device,
									device._allocator.get(),
									*transfer_command_buffer->_cmd_buffer,
									transfer_fence.get(),
									gfx_queue,
									tex_manager );
	transfer_command_buffer.reset();
	tex_manager.update_descriptors();

	OPTICK_POP();

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
		}

		auto [ image_index, swapchain_image, swapchain_image_view ] = swapchain.acquire();
		auto command_buffer = device.grab_command_buffer();
		const VkCommandBuffer cmd = *command_buffer->_cmd_buffer;

		OPTICK_GPU_CONTEXT( cmd );
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

		vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline.get() );

		command_buffer->set_viewport( draw_image_extent );
		command_buffer->set_scissor( draw_image_extent );

		vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout.get(), 0, 1, &tex_manager.m_desc, 0, nullptr );

		const glm::mat4 view = glm::translate( glm::identity<glm::mat4>(), glm::vec3 { 0, -0.9, -1.5 } );
		glm::mat4 projection = glm::perspective( glm::radians( 70.f ),
												 static_cast<float>( draw_image_extent.width ) / draw_image_extent.height,
												 10000.f,
												 0.1f );

		// invert the Y direction on projection matrix so that we are more similar
		// to opengl and gltf axis
		projection[ 1 ][ 1 ] *= -1;

		for ( const auto& mesh : meshes )
		{
			OPTICK_GPU_EVENT( "Draw Mesh" );

			const gpu_vertex_constants vertex_constants { .worldMatrix = projection * view, .vertexBuffer = mesh.vertices_address };

			vkCmdPushConstants( cmd,
								vk_pipeline_layout.get(),
								VK_SHADER_STAGE_VERTEX_BIT,
								0,
								sizeof( gpu_vertex_constants ),
								&vertex_constants );
			vkCmdPushConstants( cmd,
								vk_pipeline_layout.get(),
								VK_SHADER_STAGE_FRAGMENT_BIT,
								sizeof( gpu_vertex_constants ),
								sizeof( gpu_material ),
								&mesh.material );
			vkCmdBindIndexBuffer( cmd, mesh.indices.get(), 0, VK_INDEX_TYPE_UINT32 );

			vkCmdDrawIndexed( cmd, mesh.indices_count, 1, 0, 0, 0 );
		}

		command_buffer->end_rendering();

		command_buffer->transition_texture( draw_resolved_image, renderer::Texture::Layout::TRANSFER_SRC_OPTIMAL );
		command_buffer->transition_texture( swapchain_image, renderer::Texture::Layout::TRANSFER_DST_OPTIMAL );

		const vk::ImageBlit2 blit_region {
			.srcSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1 },
			.srcOffsets = { { vk::Offset3D {}, vk::Offset3D( draw_image._extent.width, draw_image_extent.height, 1 ) } },
			.dstSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1 },
			.dstOffsets = { { vk::Offset3D {}, vk::Offset3D( swapchain_image._extent.width, swapchain_image._extent.height, 1 ) } }
		};

		const vk::BlitImageInfo2 blit_info { .srcImage = draw_resolved_image._image,
											 .srcImageLayout = vk::ImageLayout::eTransferSrcOptimal,
											 .dstImage = swapchain_image._image,
											 .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
											 .regionCount = 1,
											 .pRegions = &blit_region };

		command_buffer->_cmd_buffer.blitImage2( blit_info );

		command_buffer->transition_texture( swapchain_image, renderer::Texture::Layout::PRESENT_SRC );

		command_buffer->end();

		swapchain.submit( *command_buffer );

		OPTICK_GPU_FLIP( *swapchain._swapchain );

		swapchain.present();

		++frame_count;
	}

	// Wait for the GPU to finish before we roll all the destructors
	vkDeviceWaitIdle( vk_device );

	OPTICK_SHUTDOWN();

	return 0;
}
