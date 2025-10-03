#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3_image/SDL_image.h>
#include <VkBootstrap.h>
#include <array>
#include <expected>
#include <fastgltf/core.hpp>
#include <filesystem>
#include <memory>
#include <optick.h>
#include <print>
#include <shaderc/shaderc.hpp>
#include <span>
#include <vma/vk_mem_alloc.h>

namespace raii
{
	struct SDL_WindowDeleter
	{
		void operator()( ::SDL_Window* window ) const { SDL_DestroyWindow( window ); }
	};
	using SDL_Window = std::unique_ptr<::SDL_Window, SDL_WindowDeleter>;

	struct SDL_SurfaceDeleter
	{
		void operator()( ::SDL_Surface* surface ) const { SDL_DestroySurface( surface ); }
	};
	using SDL_Surface = std::unique_ptr<::SDL_Surface, SDL_SurfaceDeleter>;

	struct VkSurfaceKHRDeleter
	{
		VkInstance instance;
		void operator()( ::VkSurfaceKHR surface ) const { vkDestroySurfaceKHR( instance, surface, nullptr ); }
	};
	using VkSurfaceKHR = std::unique_ptr<::VkSurfaceKHR_T, VkSurfaceKHRDeleter>;
	VkSurfaceKHR make( ::VkInstance instance, ::VkSurfaceKHR surface )
	{
		return { surface, VkSurfaceKHRDeleter( instance ) };
	}

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

	namespace details
	{
		template <class... T>
		constexpr bool always_false = false;

		template <typename T>
		void DeleteVulkanResource( T& )
		{
			static_assert( always_false<T>, "No deleter defined for type" );
		}

		template <>
		void DeleteVulkanResource( vkb::Instance& instance )
		{
			if ( instance.instance )
			{
				vkb::destroy_instance( instance );
			}
		}

		template <>
		void DeleteVulkanResource( vkb::Device& device )
		{
			if ( device.device )
			{
				vkb::destroy_device( device );
			}
		}
	}

	template <typename T>
	struct VulkanResource : T
	{
		VulkanResource() = default;
		VulkanResource( const T& res )
			: T( res )
		{
		}
		~VulkanResource() { details::DeleteVulkanResource<T>( *this ); }
		VulkanResource( const VulkanResource& ) = delete;
		VulkanResource( VulkanResource&& rhs ) noexcept
			: T( rhs )
		{
			static_cast<T&>( rhs ) = T {};
		}
		VulkanResource& operator=( const VulkanResource& ) = delete;
		VulkanResource& operator=( VulkanResource&& rhs ) noexcept
		{
			details::DeleteVulkanResource<T>( *this );
			static_cast<T&>( *this ) = rhs;
			static_cast<T&>( rhs ) = T {};
			return *this;
		}
	};

	using VulkanInstance = VulkanResource<vkb::Instance>;
	using VulkanDevice = VulkanResource<vkb::Device>;

	// Custom version for Swapchain to hold the image views
	struct VulkanSwapchain : vkb::Swapchain
	{
		std::vector<::VkImage> images;
		std::vector<::VkImageView> image_views;

		VulkanSwapchain() = default;
		VulkanSwapchain( const vkb::Swapchain& swapchain )
			: vkb::Swapchain( swapchain )
			, images( get_images().value() )
			, image_views( get_image_views().value() )
		{
		}
		~VulkanSwapchain() { free_resources(); }
		VulkanSwapchain( const VulkanSwapchain& ) = delete;
		VulkanSwapchain( VulkanSwapchain&& rhs ) noexcept
			: vkb::Swapchain( rhs )
			, images( std::move( rhs.images ) )
			, image_views( std::move( rhs.image_views ) )

		{
			static_cast<vkb::Swapchain&>( rhs ) = vkb::Swapchain {};
		}
		VulkanSwapchain& operator=( const VulkanSwapchain& ) = delete;
		VulkanSwapchain& operator=( VulkanSwapchain&& rhs ) noexcept
		{
			free_resources();
			static_cast<vkb::Swapchain&>( *this ) = rhs;
			images = std::move( rhs.images );
			image_views = std::move( rhs.image_views );
			static_cast<vkb::Swapchain&>( rhs ) = vkb::Swapchain {};
			return *this;
		}

		void free_resources()
		{
			if ( swapchain )
			{
				destroy_image_views( image_views );
				vkb::destroy_swapchain( *this );
			}
		}
	};

	struct VmaAllocatorDeleter
	{
		void operator()( ::VmaAllocator allocator ) const { vmaDestroyAllocator( allocator ); }
	};
	using VmaAllocator = std::unique_ptr<::VmaAllocator_T, VmaAllocatorDeleter>;

	struct VmaResourceDeleter
	{
		::VmaAllocator allocator;
		::VmaAllocation allocation;
		::VmaAllocationInfo info;
		void operator()( ::VkBuffer buffer ) const { vmaDestroyBuffer( allocator, buffer, allocation ); }
		void operator()( ::VkImage image ) const { vmaDestroyImage( allocator, image, allocation ); }
	};
	template <typename T>
	using VmaResource = std::unique_ptr<std::remove_pointer_t<T>, VmaResourceDeleter>;

	template <typename T>
	auto make( ::VmaAllocator allocator, T object, ::VmaAllocation allocation, const ::VmaAllocationInfo& info )
	{
		return VmaResource<T> { object, VmaResourceDeleter { allocator, allocation, info } };
	}

	using VmaBuffer = VmaResource<::VkBuffer>;
	using VmaImage = VmaResource<::VkImage>;

	struct VulkanImage
	{
		VmaImage image;
		VkImageView view;
		::VkExtent3D extent;
		::VkFormat format;
	};
}

std::expected<raii::VmaAllocator, VkResult> make_allocator( VkInstance instance, VkPhysicalDevice phys_device, VkDevice device )
{
	VmaAllocator allocator {};
	const VmaAllocatorCreateInfo allocatorInfo = {
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = phys_device,
		.device = device,
		.instance = instance,
	};
	const auto ret = vmaCreateAllocator( &allocatorInfo, &allocator );
	if ( ret )
	{
		return std::unexpected( ret );
	}
	return raii::VmaAllocator( allocator );
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

std::expected<raii::VmaBuffer, VkResult>
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

	return raii::make( allocator, buffer, allocation, info );
}

std::expected<raii::VulkanImage, VkResult> make_image( VkDevice device,
													   VmaAllocator allocator,
													   VkFormat format,
													   VkExtent3D extent,
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
								   .samples = VK_SAMPLE_COUNT_1_BIT,
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
	allocated_image.image = raii::make( allocator, image, allocation, allocation_info );
	allocated_image.extent = extent;
	allocated_image.format = format;

	const VkImageViewCreateInfo image_view_info { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
												  .image = image,
												  .viewType = VK_IMAGE_VIEW_TYPE_2D,
												  .format = format,
												  .subresourceRange = {
													  .aspectMask = aspect_flags,
													  .baseMipLevel = 0,
													  .levelCount = 1,
													  .baseArrayLayer = 0,
													  .layerCount = 1,
												  } };

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
															  .frontFace = VK_FRONT_FACE_CLOCKWISE };
	const VkPipelineMultisampleStateCreateInfo multisampling { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
															   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
															   .minSampleShading = 1.0f };
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
																  .depthCompareOp = VK_COMPARE_OP_NEVER,
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
													  .depthAttachmentFormat = VK_FORMAT_UNDEFINED };

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

std::expected<raii::VkDescriptorSetLayout, VkResult> make_descriptor_set_layout( VkDevice device )
{
	const std::array<VkDescriptorSetLayoutBinding, 2> bindings { { { .binding = 0,
																	 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
																	 .descriptorCount = 1,
																	 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
																   { .binding = 1,
																	 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
																	 .descriptorCount = 1,
																	 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT } } };

	VkDescriptorSetLayoutCreateInfo info { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
										   .bindingCount = bindings.size(),
										   .pBindings = bindings.data() };

	VkDescriptorSetLayout set = nullptr;
	const auto res = vkCreateDescriptorSetLayout( device, &info, nullptr, &set );
	if ( res )
	{
		return std::unexpected( res );
	}
	return raii::make( device, set );
}

std::expected<raii::VkDescriptorPool, VkResult> make_descriptor_pool( VkDevice device, uint32_t count )
{
	const std::array<VkDescriptorPoolSize, 1> pools { { { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
														  .descriptorCount = 2 * count } } };
	const VkDescriptorPoolCreateInfo create_info { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
												   .maxSets = count,
												   .poolSizeCount = pools.size(),
												   .pPoolSizes = pools.data() };
	VkDescriptorPool pool = nullptr;
	const auto res = vkCreateDescriptorPool( device, &create_info, nullptr, &pool );
	if ( res )
	{
		return std::unexpected( res );
	}
	return raii::make( device, pool );
}

void update_descriptors( VkDevice device, VkDescriptorSet set, VkSampler sampler, VkImageView image1, VkImageView image2 )
{
	const std::array<VkDescriptorImageInfo, 2> infos {
		{ { .sampler = sampler, .imageView = image1, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
		  { .sampler = sampler, .imageView = image2, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL } }
	};

	const std::array<VkWriteDescriptorSet, 2> writes { { { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
														   .dstSet = set,
														   .dstBinding = 0,
														   .descriptorCount = 1,
														   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
														   .pImageInfo = &infos[ 0 ] },
														 { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
														   .dstSet = set,
														   .dstBinding = 1,
														   .descriptorCount = 1,
														   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
														   .pImageInfo = &infos[ 1 ] } } };

	vkUpdateDescriptorSets( device, writes.size(), writes.data(), 0, nullptr );
}

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
	raii::SDL_Surface image;
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
};

struct mesh_vertex
{
	float position[ 3 ];
	float u;
	float normal[ 3 ];
	float v;
	float tangent[ 4 ];
};

struct gpu_mesh_data
{
	raii::VmaBuffer vertices;
	raii::VmaBuffer indices;
	raii::VulkanImage albedo_texture;
	raii::VulkanImage normal_texture;
	VkDeviceAddress vertices_address;
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

VkFormat sdl_to_vulkan_format( SDL_PixelFormat format )
{
	switch ( format )
	{
		case SDL_PIXELFORMAT_BGR565:
			return VK_FORMAT_B5G6R5_UNORM_PACK16;
		case SDL_PIXELFORMAT_BGRA5551:
			return VK_FORMAT_B5G5R5A1_UNORM_PACK16;
		case SDL_PIXELFORMAT_BGRA32:
		case SDL_PIXELFORMAT_BGRX32:
			return VK_FORMAT_B8G8R8A8_UNORM;
		case SDL_PIXELFORMAT_RGBA32:
		case SDL_PIXELFORMAT_RGBX32:
			return VK_FORMAT_R8G8B8A8_UNORM;
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

std::expected<raii::VulkanImage, VkResult> make_image_from_surface( VkDevice device, VmaAllocator allocator, const SDL_Surface& surface )
{
	OPTICK_EVENT();
	const auto format = sdl_to_vulkan_format( sdl_gpu_format( surface.format ) );
	if ( format == VK_FORMAT_UNDEFINED )
	{
		return std::unexpected( VK_ERROR_FORMAT_NOT_SUPPORTED );
	}
	return make_image( device,
					   allocator,
					   format,
					   VkExtent3D( surface.w, surface.h, 1 ),
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

std::vector<gpu_mesh_data> load_model( VkDevice device, VmaAllocator allocator, VkCommandBuffer cmd, VkFence fence, VkQueue queue )
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
														   .buffer = gpu_data.vertices.get() };
		gpu_data.vertices_address = vkGetBufferDeviceAddress( device, &deviceAdressInfo );

		gpu_data.indices = make_buffer( allocator,
										sizeof( uint32_t ) * mesh.indices.length,
										VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
										VMA_MEMORY_USAGE_GPU_ONLY )
							   .value();

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
		vkCmdCopyBuffer( cmd, transfer_buffer.get(), gpu_data.vertices.get(), 1, &vertexCopy );
		offset += sizeof( mesh_vertex ) * mesh.positions.length;

		memcpy( ptr + offset, gltf_buffer.data() + mesh.indices.offset, sizeof( uint32_t ) * mesh.indices.length );
		const VkBufferCopy indexCopy { .srcOffset = offset, .dstOffset = 0, .size = sizeof( uint32_t ) * mesh.indices.length };
		vkCmdCopyBuffer( cmd, transfer_buffer.get(), gpu_data.indices.get(), 1, &indexCopy );
		offset += sizeof( uint32_t ) * mesh.indices.length;

		gpu_data.albedo_texture = *make_image_from_surface( device, allocator, *mesh.albedo.image );
		offset += upload_image_from_surface( cmd,
											 transfer_buffer.get(),
											 ptr,
											 offset,
											 gpu_data.albedo_texture.image.get(),
											 *mesh.albedo.image );

		gpu_data.normal_texture = *make_image_from_surface( device, allocator, *mesh.normal.image );
		offset += upload_image_from_surface( cmd,
											 transfer_buffer.get(),
											 ptr,
											 offset,
											 gpu_data.normal_texture.image.get(),
											 *mesh.normal.image );
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

int main()
{
	constexpr bool profile_startup = false;

	if ( profile_startup )
	{
		OPTICK_START_CAPTURE();
	}

	OPTICK_PUSH( "Startup" );

	{
		OPTICK_EVENT( "SDL_init" );
		if ( !SDL_Init( SDL_INIT_VIDEO ) )
		{
			std::println( stderr, "Failed to initialize SDL: {}", SDL_GetError() );
			return 1;
		}
	}

	const VkExtent2D window_extent { .width = 1024, .height = 768 };
	const raii::SDL_Window window(
		SDL_CreateWindow( "Brutus-ng Vulkan", window_extent.width, window_extent.height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN ) );
	if ( !window )
	{
		std::println( stderr, "Failed to create window: {}", SDL_GetError() );
		return 1;
	}

	raii::VulkanInstance vk_instance;
	{
		OPTICK_EVENT( "create_vk_instance" );
		Uint32 ext_count = 0;
		const auto exts = SDL_Vulkan_GetInstanceExtensions( &ext_count );

		const auto vk_instance_result = vkb::InstanceBuilder()
											.set_app_name( "Brutus-ng Vulkan" )
											.request_validation_layers( true )
											.use_default_debug_messenger()
											.require_api_version( 1, 3, 0 )
											.enable_extensions( ext_count, exts )
											.build();

		if ( !vk_instance_result )
		{
			std::println( stderr, "Failed to init vulkan: {}", vk_instance_result.error().message() );
			return 1;
		}

		vk_instance = vk_instance_result.value();
	}

	raii::VkSurfaceKHR vk_surface;
	{
		OPTICK_EVENT( "create_vk_surface" );
		VkSurfaceKHR tmp {};
		if ( !SDL_Vulkan_CreateSurface( window.get(), vk_instance.instance, nullptr, &tmp ) )
		{
			std::println( stderr, "Failed to create SDL vulkan surface: {}", SDL_GetError() );
			return 1;
		}
		vk_surface = raii::make( vk_instance, tmp );
	}

	raii::VulkanDevice vk_device;
	{
		OPTICK_EVENT( "create_vk_device" );
		const VkPhysicalDeviceVulkan13Features features13 { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
															.synchronization2 = true,
															.dynamicRendering = true };

		const VkPhysicalDeviceVulkan12Features features12 { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
															.descriptorIndexing = true,
															.bufferDeviceAddress = true };

		const auto physical_device_ret = vkb::PhysicalDeviceSelector( vk_instance )
											 .set_minimum_version( 1, 3 )
											 .set_required_features_13( features13 )
											 .set_required_features_12( features12 )
											 .set_surface( vk_surface.get() )
											 .select();

		if ( !physical_device_ret )
		{
			std::println( stderr, "Cannot find a suitable hardware device: {}", physical_device_ret.error().message() );
			return 1;
		}

		const auto device_ret = vkb::DeviceBuilder( physical_device_ret.value() ).build();
		if ( !device_ret )
		{
			std::println( stderr, "Cannot create hardware device: {}", device_ret.error().message() );
			return 1;
		}

		vk_device = device_ret.value();

		VkPhysicalDeviceProperties2 props { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
		vkGetPhysicalDeviceProperties2( vk_device.physical_device, &props );

		std::println( "Using hardware device: {}", props.properties.deviceName );
	}

	constexpr auto swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;
	raii::VulkanSwapchain vk_swapchain;
	{
		OPTICK_EVENT( "create_vk_swapchain" );
		const auto swapchain_ret = vkb::SwapchainBuilder( vk_device )
									   .set_desired_format( VkSurfaceFormatKHR { .format = swapchain_image_format,
																				 .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR } )
									   .set_desired_present_mode( VK_PRESENT_MODE_FIFO_KHR )
									   .set_desired_extent( window_extent.width, window_extent.height )
									   .add_image_usage_flags( VK_IMAGE_USAGE_TRANSFER_DST_BIT )
									   .build();
		if ( !swapchain_ret )
		{
			std::println( stderr, "Cannot create hardware device: {}", swapchain_ret.error().message() );
			return 1;
		}

		vk_swapchain = swapchain_ret.value();
	}

	const auto vk_allocator = make_allocator( vk_instance, vk_device.physical_device, vk_device ).value();

	const VkExtent2D draw_image_extent = window_extent;
	const auto draw_image = make_image( vk_device,
										vk_allocator.get(),
										VK_FORMAT_R16G16B16A16_SFLOAT,
										VkExtent3D { .width = draw_image_extent.width, .height = draw_image_extent.height, .depth = 1 },
										VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT
											| VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
										VMA_MEMORY_USAGE_GPU_ONLY,
										VK_IMAGE_ASPECT_COLOR_BIT )
								.value();

	const auto vk_gfx_queue = vk_device.get_queue( vkb::QueueType::graphics ).value();
	const auto vk_gfx_queue_family = vk_device.get_queue_index( vkb::QueueType::graphics ).value();

	raii::VkCommandPool vk_command_pool;
	{
		const VkCommandPoolCreateInfo commandPoolInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
														  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
														  .queueFamilyIndex = vk_gfx_queue_family };
		VkCommandPool pool;
		const auto ret = vkCreateCommandPool( vk_device, &commandPoolInfo, nullptr, &pool );
		if ( ret )
		{
			std::println( stderr, "Cannot create command pool: {}", static_cast<int>( ret ) );
			return 1;
		}
		vk_command_pool = raii::make( vk_device, pool );
	}

	struct FrameData
	{
		VkCommandBuffer command_buffer;
		raii::VkFence render_fence;
		raii::VkSemaphore render_semaphore;
		raii::VkSemaphore swapchain_semaphore;
	};

	std::array<FrameData, 2> vk_frame_data;
	for ( int i = 0; i < vk_frame_data.size(); ++i )
	{
		const VkCommandBufferAllocateInfo cmdAllocInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
														   .commandPool = vk_command_pool.get(),
														   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
														   .commandBufferCount = 1 };

		auto ret = vkAllocateCommandBuffers( vk_device, &cmdAllocInfo, &vk_frame_data[ i ].command_buffer );
		if ( ret )
		{
			std::println( stderr, "Cannot allocate command buffer: {}", static_cast<int>( ret ) );
			return 1;
		}
		vk_frame_data[ i ].render_fence = make_fence( vk_device, VK_FENCE_CREATE_SIGNALED_BIT ).value();
		vk_frame_data[ i ].render_semaphore = make_semaphore( vk_device ).value();
		vk_frame_data[ i ].swapchain_semaphore = make_semaphore( vk_device ).value();
	}

	const auto vk_descriptor_set_layout = *make_descriptor_set_layout( vk_device );
	const auto vk_descriptor_pool = *make_descriptor_pool( vk_device, 256 );
	const auto linear_sampler = *make_sampler( vk_device, VK_FILTER_LINEAR );

	raii::VkPipelineLayout vk_pipeline_layout;
	raii::VkPipeline vk_pipeline;
	{
		OPTICK_EVENT( "create_vk_pipeline" );
		VkPipelineLayout layout {};
		const auto desc_layout = vk_descriptor_set_layout.get();
		VkPipelineLayoutCreateInfo pipeline_layout_info { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
														  .setLayoutCount = 1,
														  .pSetLayouts = &desc_layout };
		const auto ret = vkCreatePipelineLayout( vk_device, &pipeline_layout_info, nullptr, &layout );
		if ( ret )
		{
			std::println( stderr, "Cannot create pipeline layout: {}", static_cast<int>( ret ) );
			return 1;
		}
		vk_pipeline_layout = raii::make( vk_device, layout );

		const auto vertex_shader = make_shader( vk_device, "../shaders/vulkan/triangle.vs", VK_SHADER_STAGE_VERTEX_BIT ).value();
		const auto fragment_shader = make_shader( vk_device, "../shaders/vulkan/triangle.fs", VK_SHADER_STAGE_FRAGMENT_BIT ).value();

		vk_pipeline = make_pipeline( vk_device, vk_swapchain.image_format, layout, vertex_shader.get(), fragment_shader.get() ).value();
	}

	VkCommandBuffer transfer_command_buffer {};
	{
		const VkCommandBufferAllocateInfo cmdAllocInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
														   .commandPool = vk_command_pool.get(),
														   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
														   .commandBufferCount = 1 };

		vkAllocateCommandBuffers( vk_device, &cmdAllocInfo, &transfer_command_buffer );
	}
	const auto transfer_fence = make_fence( vk_device, 0 ).value();

	const auto meshes = load_model( vk_device, vk_allocator.get(), transfer_command_buffer, transfer_fence.get(), vk_gfx_queue );
	std::vector<VkDescriptorSet> mesh_sets;
	for ( const auto& mesh : meshes )
	{
		const auto layout = vk_descriptor_set_layout.get();
		const VkDescriptorSetAllocateInfo alloc_info { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
													   .descriptorPool = vk_descriptor_pool.get(),
													   .descriptorSetCount = 1,
													   .pSetLayouts = &layout };
		auto& set = mesh_sets.emplace_back( nullptr );
		vkAllocateDescriptorSets( vk_device, &alloc_info, &set );
		update_descriptors( vk_device, set, linear_sampler.get(), mesh.albedo_texture.view.get(), mesh.normal_texture.view.get() );
	}

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

		auto& frame_data = vk_frame_data[ frame_count % 2 ];

		const auto fence = frame_data.render_fence.get();
		vkWaitForFences( vk_device, 1, &fence, true, 1000000000 );
		vkResetFences( vk_device, 1, &fence );

		uint32_t swapchainImageIndex;
		vkAcquireNextImageKHR( vk_device, vk_swapchain, 1000000000, frame_data.render_semaphore.get(), nullptr, &swapchainImageIndex );

		vkResetCommandBuffer( frame_data.command_buffer, 0 );

		const VkCommandBufferBeginInfo cmdBeginInfo { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
													  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		vkBeginCommandBuffer( frame_data.command_buffer, &cmdBeginInfo );

		transition_image( frame_data.command_buffer, draw_image.image.get(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL );

		VkClearColorValue clearValue;
		float flash = std::abs( std::sin( frame_count / 120.f ) );
		clearValue = { { 0.0f, 0.0f, flash, 1.0f } };

		const VkImageSubresourceRange clearRange { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
												   .levelCount = VK_REMAINING_MIP_LEVELS,
												   .layerCount = VK_REMAINING_ARRAY_LAYERS };
		vkCmdClearColorImage( frame_data.command_buffer, draw_image.image.get(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange );

		const VkRenderingAttachmentInfo color_attachment {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = draw_image.view.get(),
			.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		};

		const VkRenderingInfo renderInfo { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
										   .renderArea = VkRect2D { VkOffset2D { 0, 0 }, draw_image_extent },
										   .layerCount = 1,
										   .colorAttachmentCount = 1,
										   .pColorAttachments = &color_attachment };

		vkCmdBeginRendering( frame_data.command_buffer, &renderInfo );

		vkCmdBindPipeline( frame_data.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline.get() );
		vkCmdBindDescriptorSets( frame_data.command_buffer,
								 VK_PIPELINE_BIND_POINT_GRAPHICS,
								 vk_pipeline_layout.get(),
								 0,
								 1,
								 &mesh_sets[ frame_count % mesh_sets.size() ],
								 0,
								 nullptr );

		const VkViewport viewport { .x = 0.f,
									.y = 0.f,
									.width = static_cast<float>( draw_image_extent.width ),
									.height = static_cast<float>( draw_image_extent.height ),
									.minDepth = 0.f,
									.maxDepth = 1.f };

		vkCmdSetViewport( frame_data.command_buffer, 0, 1, &viewport );

		const VkRect2D scissor { .extent = draw_image_extent };
		vkCmdSetScissor( frame_data.command_buffer, 0, 1, &scissor );

		vkCmdDraw( frame_data.command_buffer, 3, 1, 0, 0 );

		vkCmdEndRendering( frame_data.command_buffer );

		transition_image( frame_data.command_buffer,
						  draw_image.image.get(),
						  VK_IMAGE_LAYOUT_GENERAL,
						  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL );
		transition_image( frame_data.command_buffer,
						  vk_swapchain.images[ swapchainImageIndex ],
						  VK_IMAGE_LAYOUT_UNDEFINED,
						  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );

		blit_image( frame_data.command_buffer,
					draw_image.image.get(),
					vk_swapchain.images[ swapchainImageIndex ],
					VkExtent2D { draw_image.extent.width, draw_image.extent.height },
					vk_swapchain.extent );

		transition_image( frame_data.command_buffer,
						  vk_swapchain.images[ swapchainImageIndex ],
						  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
						  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR );

		vkEndCommandBuffer( frame_data.command_buffer );

		const VkCommandBufferSubmitInfo cmdSubmitInfo { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
														.commandBuffer = frame_data.command_buffer };

		const VkSemaphoreSubmitInfo waitInfo { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
											   .semaphore = frame_data.swapchain_semaphore.get(),
											   .value = 1,
											   .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR };

		const VkSemaphoreSubmitInfo signalInfo { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
												 .semaphore = frame_data.render_semaphore.get(),
												 .value = 1,
												 .stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT };

		const VkSubmitInfo2 submitInfo {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.waitSemaphoreInfoCount = 1,
			.pWaitSemaphoreInfos = &waitInfo,
			.commandBufferInfoCount = 1,
			.pCommandBufferInfos = &cmdSubmitInfo,
			.signalSemaphoreInfoCount = 1,
			.pSignalSemaphoreInfos = &signalInfo,
		};
		vkQueueSubmit2( vk_gfx_queue, 1, &submitInfo, frame_data.render_fence.get() );

		const auto semaphore = frame_data.render_semaphore.get();
		const VkPresentInfoKHR presentInfo = {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &semaphore,
			.swapchainCount = 1,
			.pSwapchains = &vk_swapchain.swapchain,
			.pImageIndices = &swapchainImageIndex,
		};
		vkQueuePresentKHR( vk_gfx_queue, &presentInfo );

		++frame_count;
	}

	// Wait for the GPU to finish before we roll all the destructors
	vkDeviceWaitIdle( vk_device );

	return 0;
}
