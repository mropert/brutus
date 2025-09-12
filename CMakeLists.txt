cmake_minimum_required(VERSION 3.18)
project(brutus-ng CXX)

find_package(raylib CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)

add_compile_options(/Zi /MP)
add_link_options(/DEBUG)

add_library(utils INTERFACE)
target_compile_features(utils INTERFACE cxx_std_20)
target_include_directories(utils INTERFACE src)
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../raylib-cpp/include)
target_link_libraries(utils INTERFACE raylib)

if(USE_OPTICK)
	target_compile_definitions(utils INTERFACE USE_OPTICK)
	target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../optick/include)
	target_link_libraries(utils INTERFACE
		debug ${CMAKE_CURRENT_SOURCE_DIR}/../optick/lib/x64/debug/OptickCore.lib
		optimized ${CMAKE_CURRENT_SOURCE_DIR}/../optick/lib/x64/release/OptickCore.lib)
endif()

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
target_compile_features(brutus-sdl PRIVATE cxx_std_23)
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

add_library(VkBoostrap STATIC)
target_compile_features(VkBoostrap PRIVATE cxx_std_23)
target_sources(VkBoostrap PRIVATE third_party/vk-bootstrap/v1.4.305/VkBootstrap.cpp)
target_include_directories(VkBoostrap PUBLIC third_party/vk-bootstrap/v1.4.305)
target_link_libraries(VkBoostrap PUBLIC Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)

add_executable(brutus-vulkan)
target_compile_features(brutus-vulkan PRIVATE cxx_std_23)
target_sources(brutus-vulkan PRIVATE vulkan_prototype.cpp vma_impl.cpp)
target_link_libraries(brutus-vulkan PRIVATE SDL3::SDL3 Vulkan::Vulkan VkBoostrap unofficial::shaderc::shaderc)
if(USE_OPTICK)
	target_compile_definitions(brutus-vulkan PRIVATE USE_OPTICK)
	target_include_directories(brutus-vulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../optick/include)
	target_link_libraries(brutus-vulkan PRIVATE
		debug ${CMAKE_CURRENT_SOURCE_DIR}/../optick/lib/x64/debug/OptickCore.lib
		optimized ${CMAKE_CURRENT_SOURCE_DIR}/../optick/lib/x64/release/OptickCore.lib)
endif()

enable_testing()

add_executable(tests)
target_sources(tests
	PRIVATE
		src/tests/bitset_tests.cpp
)
target_link_libraries(tests PRIVATE utils)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
