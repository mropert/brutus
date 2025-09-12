#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
#include <array>
#include <expected>
#include <memory>
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
		void operator()( ::VkFence fence ) const { vkDestroyFence( device, fence, nullptr ); }
		void operator()( ::VkPipeline pipeline ) const { vkDestroyPipeline( device, pipeline, nullptr ); }
		void operator()( ::VkPipelineLayout layout ) const { vkDestroyPipelineLayout( device, layout, nullptr ); }
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
	using VkFence = VkObject<::VkFence>;
	using VkPipeline = VkObject<::VkPipeline>;
	using VkPipelineLayout = VkObject<::VkPipelineLayout>;
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

std::expected<raii::VkShaderModule, VkResult>
make_shader( VkDevice device, const std::string_view source, VkShaderStageFlagBits type, const char* filename )
{
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

std::expected<raii::VkPipeline, VkResult> make_pipeline( VkDevice device,
														 VkFormat color_format,
														 VkPipelineLayout layout,
														 VkShaderModule vertex_shader,
														 VkShaderModule fragment_shader )
{
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

static constexpr std::string_view vertex_shader_code = R"(#version 450

layout( location = 0 ) out vec3 outColor;

void main()
{
	const vec3 positions[ 3 ] = vec3[ 3 ]( vec3( 1.f, 1.f, 0.0f ), vec3( -1.f, 1.f, 0.0f ), vec3( 0.f, -1.f, 0.0f ) );

	const vec3 colors[ 3 ] = vec3[ 3 ]( vec3( 1.0f, 0.0f, 0.0f ), //red
										vec3( 0.0f, 1.0f, 0.0f ), //green
										vec3( 00.f, 0.0f, 1.0f ) //blue
	);

	gl_Position = vec4( positions[ gl_VertexIndex ], 1.0f );
	outColor = colors[ gl_VertexIndex ];
})";

static constexpr std::string_view fragment_shader_code = R"(#version 450

layout (location = 0) in vec3 inColor;
layout (location = 0) out vec4 outFragColor;

void main() 
{
	//return red
	outFragColor = vec4(inColor,1.0f);
})";

int main()
{
	if ( !SDL_Init( SDL_INIT_VIDEO ) )
	{
		std::println( stderr, "Failed to initialize SDL: {}", SDL_GetError() );
		return 1;
	}

	const raii::SDL_Window window( SDL_CreateWindow( "Brutus-ng Vulkan", 1024, 768, SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN ) );
	if ( !window )
	{
		std::println( stderr, "Failed to create window: {}", SDL_GetError() );
		return 1;
	}

	raii::VulkanInstance vk_instance;
	{
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
		const auto swapchain_ret = vkb::SwapchainBuilder( vk_device )
									   .set_desired_format( VkSurfaceFormatKHR { .format = swapchain_image_format,
																				 .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR } )
									   .set_desired_present_mode( VK_PRESENT_MODE_FIFO_KHR )
									   .set_desired_extent( 1024, 768 )
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

		const VkFenceCreateInfo fenceCreateInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
		VkFence fence;
		ret = vkCreateFence( vk_device, &fenceCreateInfo, nullptr, &fence );
		if ( ret )
		{
			std::println( stderr, "Cannot create fence: {}", static_cast<int>( ret ) );
			return 1;
		}
		vk_frame_data[ i ].render_fence = raii::make( vk_device, fence );

		const VkSemaphoreCreateInfo semaphoreCreateInfo = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		VkSemaphore semaphore;
		ret = vkCreateSemaphore( vk_device, &semaphoreCreateInfo, nullptr, &semaphore );
		if ( ret )
		{
			std::println( stderr, "Cannot create semaphore: {}", static_cast<int>( ret ) );
			return 1;
		}
		vk_frame_data[ i ].render_semaphore = raii::make( vk_device, semaphore );

		ret = vkCreateSemaphore( vk_device, &semaphoreCreateInfo, nullptr, &semaphore );
		if ( ret )
		{
			std::println( stderr, "Cannot create semaphore: {}", static_cast<int>( ret ) );
			return 1;
		}
		vk_frame_data[ i ].swapchain_semaphore = raii::make( vk_device, semaphore );
	}

	raii::VkPipelineLayout vk_pipeline_layout;
	raii::VkPipeline vk_pipeline;
	{
		VkPipelineLayout layout {};
		VkPipelineLayoutCreateInfo pipeline_layout_info { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		const auto ret = vkCreatePipelineLayout( vk_device, &pipeline_layout_info, nullptr, &layout );
		if ( ret )
		{
			std::println( stderr, "Cannot create pipeline layout: {}", static_cast<int>( ret ) );
			return 1;
		}
		vk_pipeline_layout = raii::make( vk_device, layout );

		const auto vertex_shader = make_shader( vk_device, vertex_shader_code, VK_SHADER_STAGE_VERTEX_BIT, "triangle.vs" ).value();
		const auto fragment_shader = make_shader( vk_device, fragment_shader_code, VK_SHADER_STAGE_FRAGMENT_BIT, "triangle.fs" ).value();

		vk_pipeline = make_pipeline( vk_device, vk_swapchain.image_format, layout, vertex_shader.get(), fragment_shader.get() ).value();
	}

	bool quit = false;
	int frame_count = 0;

	while ( !quit )
	{
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

		transition_image( frame_data.command_buffer,
						  vk_swapchain.images[ swapchainImageIndex ],
						  VK_IMAGE_LAYOUT_UNDEFINED,
						  VK_IMAGE_LAYOUT_GENERAL );

		VkClearColorValue clearValue;
		float flash = std::abs( std::sin( frame_count / 120.f ) );
		clearValue = { { 0.0f, 0.0f, flash, 1.0f } };

		const VkImageSubresourceRange clearRange { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
												   .levelCount = VK_REMAINING_MIP_LEVELS,
												   .layerCount = VK_REMAINING_ARRAY_LAYERS };
		vkCmdClearColorImage( frame_data.command_buffer,
							  vk_swapchain.images[ swapchainImageIndex ],
							  VK_IMAGE_LAYOUT_GENERAL,
							  &clearValue,
							  1,
							  &clearRange );

		const VkRenderingAttachmentInfo color_attachment {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = vk_swapchain.image_views[ swapchainImageIndex ],
			.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		};

		const VkRenderingInfo renderInfo { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
										   .renderArea = VkRect2D { VkOffset2D { 0, 0 }, vk_swapchain.extent },
										   .layerCount = 1,
										   .colorAttachmentCount = 1,
										   .pColorAttachments = &color_attachment };

		vkCmdBeginRendering( frame_data.command_buffer, &renderInfo );

		vkCmdBindPipeline( frame_data.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline.get() );

		const VkViewport viewport { .x = 0.f,
									.y = 0.f,
									.width = static_cast<float>( vk_swapchain.extent.width ),
									.height = static_cast<float>( vk_swapchain.extent.height ),
									.minDepth = 0.f,
									.maxDepth = 1.f };

		vkCmdSetViewport( frame_data.command_buffer, 0, 1, &viewport );

		const VkRect2D scissor { .extent = vk_swapchain.extent };
		vkCmdSetScissor( frame_data.command_buffer, 0, 1, &scissor );

		vkCmdDraw( frame_data.command_buffer, 3, 1, 0, 0 );

		vkCmdEndRendering( frame_data.command_buffer );

		transition_image( frame_data.command_buffer,
						  vk_swapchain.images[ swapchainImageIndex ],
						  VK_IMAGE_LAYOUT_GENERAL,
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
