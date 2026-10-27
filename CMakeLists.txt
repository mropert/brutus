cmake_minimum_required(VERSION 3.18)

option(USE_TRACY "Enable Tracy profiler instrumentation" OFF)
if(USE_TRACY)
	list(APPEND VCPKG_MANIFEST_FEATURES "tracy")
endif()

project(brutus-ng CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
	add_compile_options(/Zi /MP /W4 /WX)
	add_link_options(/DEBUG)

	add_library(Optick INTERFACE)
	set(OPTICK_FOUND TRUE)
	set(OPTICK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../optick)
	target_compile_definitions(Optick INTERFACE USE_OPTICK)
	target_include_directories(Optick INTERFACE ${OPTICK_DIR}/include)
	target_link_libraries(Optick INTERFACE
	debug ${OPTICK_DIR}/lib/x64/debug/OptickCore.lib
	optimized ${OPTICK_DIR}/lib/x64/release/OptickCore.lib)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	add_compile_options(-W -Wall -Werror)
endif()

add_subdirectory(submodules/vk-renderer)
add_subdirectory(submodules/gpu_stable_power)

find_package(fastgltf CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(mikktspace CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)

if(USE_TRACY)
	find_package(Tracy CONFIG REQUIRED)
endif()

add_executable(brutus-vulkan)
target_sources(brutus-vulkan PRIVATE
		src/asset.cpp
		src/camera.cpp
		src/main.cpp
		src/scene_renderer.cpp
)
target_compile_definitions(brutus-vulkan PRIVATE GLM_FORCE_INTRINSICS GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_link_libraries(brutus-vulkan PRIVATE
	argparse::argparse
	fastgltf::fastgltf
	glm::glm-header-only
	gpu_stable_power
	imgui::imgui
	meshoptimizer::meshoptimizer
	mikktspace::mikktspace
	renderer
	SDL3::SDL3
	TBB::tbb
)
target_link_libraries(brutus-vulkan PRIVATE $<IF:$<TARGET_EXISTS:SDL3_image::SDL3_image-shared>,SDL3_image::SDL3_image-shared,SDL3_image::SDL3_image-static>)

if(OPTICK_FOUND)
	target_link_libraries(brutus-vulkan PRIVATE Optick)
endif()

if(USE_TRACY)
	target_compile_definitions(brutus-vulkan PRIVATE USE_TRACY)
	target_link_libraries(brutus-vulkan PRIVATE Tracy::TracyClient)
endif()

set_target_properties(brutus-vulkan PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(brutus-vulkan PRIVATE "/d1trimfile:${CMAKE_SOURCE_DIR}\\")
endif()