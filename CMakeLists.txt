cmake_minimum_required(VERSION 3.18)
project(brutus-ng CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(raylib CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)

add_compile_options(/Zi /MP)
add_link_options(/DEBUG)

add_library(Optick INTERFACE)
if(USE_OPTICK)
	set(OPTICK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../optick)
	target_compile_definitions(Optick INTERFACE USE_OPTICK)
	target_include_directories(Optick INTERFACE ${OPTICK_DIR}/include)
	target_link_libraries(Optick INTERFACE
		debug ${OPTICK_DIR}/lib/x64/debug/OptickCore.lib
		optimized ${OPTICK_DIR}/lib/x64/release/OptickCore.lib)
endif()

add_library(utils INTERFACE)
target_include_directories(utils INTERFACE src)
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../raylib-cpp/include)
target_link_libraries(utils INTERFACE raylib Optick)

add_executable(brutus-ng)
target_sources(brutus-ng
	PRIVATE
		main.cpp
		src/game/camera.cpp
		src/game/gamestate.cpp
		src/game/gfx.cpp
		src/game/gfx_map.cpp
		src/game/map.cpp
		src/game/pathfinding.cpp
)

target_link_libraries(brutus-ng PRIVATE utils)


find_package(SDL3 CONFIG REQUIRED)
find_package(unofficial-shaderc CONFIG REQUIRED)

add_executable(brutus-sdl)
target_sources(brutus-sdl PRIVATE sdl_test.cpp)
target_link_libraries(brutus-sdl PRIVATE SDL3::SDL3 unofficial::shaderc::shaderc)
if(USE_OPTICK)
	target_compile_definitions(brutus-sdl PRIVATE USE_OPTICK)
	target_include_directories(brutus-sdl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../optick/include)
	target_link_libraries(brutus-sdl PRIVATE
		debug ${CMAKE_CURRENT_SOURCE_DIR}/../optick/lib/x64/debug/OptickCore.lib
		optimized ${CMAKE_CURRENT_SOURCE_DIR}/../optick/lib/x64/release/OptickCore.lib)
endif()

find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(fastgltf CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_library(VkBoostrap STATIC)
target_sources(VkBoostrap PRIVATE third_party/vk-bootstrap/v1.4.305/VkBootstrap.cpp)
target_include_directories(VkBoostrap PUBLIC third_party/vk-bootstrap/v1.4.305)
target_link_libraries(VkBoostrap PUBLIC Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)

add_library( VulkanHppModule )
target_sources( VulkanHppModule PRIVATE
  FILE_SET CXX_MODULES
  BASE_DIRS ${Vulkan_INCLUDE_DIR}
  FILES ${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
)
target_compile_definitions( VulkanHppModule PUBLIC
  VULKAN_HPP_NO_SETTERS
  VULKAN_HPP_NO_CONSTRUCTORS
)
target_link_libraries( VulkanHppModule PUBLIC Vulkan::Vulkan )

add_library(renderer)
target_sources(renderer
	PRIVATE
		src/renderer/command_buffer.cpp
		src/renderer/device.cpp
		src/renderer/swapchain.cpp
		src/renderer/texture.cpp
		src/renderer/vma_impl.cpp
)
target_include_directories(renderer PUBLIC src)
target_link_libraries(renderer PRIVATE SDL3::SDL3 Vulkan::Vulkan VkBoostrap unofficial::shaderc::shaderc fastgltf::fastgltf)
target_link_libraries(renderer PRIVATE $<IF:$<TARGET_EXISTS:SDL3_image::SDL3_image-shared>,SDL3_image::SDL3_image-shared,SDL3_image::SDL3_image-static>)
target_link_libraries(renderer PUBLIC VulkanHppModule)
target_link_libraries(renderer PRIVATE Optick)

add_executable(brutus-vulkan)
target_sources(brutus-vulkan PRIVATE vulkan_prototype.cpp)
target_compile_definitions(brutus-vulkan PRIVATE GLM_FORCE_INTRINSICS GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_link_libraries(brutus-vulkan PRIVATE SDL3::SDL3 Vulkan::Vulkan VkBoostrap unofficial::shaderc::shaderc fastgltf::fastgltf glm::glm-header-only)
target_link_libraries(brutus-vulkan PRIVATE $<IF:$<TARGET_EXISTS:SDL3_image::SDL3_image-shared>,SDL3_image::SDL3_image-shared,SDL3_image::SDL3_image-static>)
target_link_libraries(brutus-vulkan PRIVATE renderer)
target_link_libraries(brutus-vulkan PRIVATE Optick)

set_target_properties(brutus-vulkan PROPERTIES
    VS_DEBUGGER_ENVIRONMENT "VK_ADD_LAYER_PATH=${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
)

enable_testing()

add_executable(tests)
target_sources(tests
	PRIVATE
		src/tests/bitset_tests.cpp
)
target_link_libraries(tests PRIVATE utils)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
