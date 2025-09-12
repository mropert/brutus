#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
#include <array>
#include <memory>
#include <print>

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
		void operator()( ::VkSemaphore semaphore ) const { vkDestroySemaphore( device, semaphore, nullptr ); }
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
	using VkSemaphore = VkObject<::VkSemaphore>;

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
			std::println( stderr,
						  "Failed to create SDL vulkan "
						  "surface: {}",
						  SDL_GetError() );
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
			std::println( stderr,
						  "Cannot find a suitable "
						  "hardware device: {}",
						  physical_device_ret.error().message() );
			return 1;
		}

		const auto device_ret = vkb::DeviceBuilder( physical_device_ret.value() ).build();
		if ( !device_ret )
		{
			std::println( stderr,
						  "Cannot create hardware "
						  "device: {}",
						  device_ret.error().message() );
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
			std::println( stderr,
						  "Cannot create hardware "
						  "device: {}",
						  swapchain_ret.error().message() );
			return 1;
		}

		vk_swapchain = swapchain_ret.value();
	}

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
			std::println( stderr,
						  "Cannot create command pool: "
						  "{}",
						  static_cast<int>( ret ) );
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
			std::println( stderr,
						  "Cannot allocate command "
						  "buffer: {}",
						  static_cast<int>( ret ) );
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
