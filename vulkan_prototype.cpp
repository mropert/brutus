#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
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
		std::vector<VkImageView> image_views;

		VulkanSwapchain() = default;
		VulkanSwapchain( const vkb::Swapchain& swapchain )
			: vkb::Swapchain( swapchain )
			, image_views( get_image_views().value() )
		{
		}
		~VulkanSwapchain() { free_resources(); }
		VulkanSwapchain( const VulkanSwapchain& ) = delete;
		VulkanSwapchain( VulkanSwapchain&& rhs ) noexcept
			: vkb::Swapchain( rhs )
		{
			static_cast<vkb::Swapchain&>( rhs ) = vkb::Swapchain {};
		}
		VulkanSwapchain& operator=( const VulkanSwapchain& ) = delete;
		VulkanSwapchain& operator=( VulkanSwapchain&& rhs ) noexcept
		{
			free_resources();
			static_cast<vkb::Swapchain&>( *this ) = rhs;
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
		vk_surface = { tmp, raii::VkSurfaceKHRDeleter{ vk_instance.instance } };
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
	raii::VulkanSwapchain swapchain;
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

		swapchain = swapchain_ret.value();
	}

	bool quit = false;

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
	}

	return 0;
}
