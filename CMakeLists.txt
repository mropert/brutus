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

add_executable(brutus-vulkan)
target_sources(brutus-vulkan PRIVATE vulkan_prototype.cpp asset.cpp)
target_compile_definitions(brutus-vulkan PRIVATE GLM_FORCE_INTRINSICS GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_link_libraries(brutus-vulkan PRIVATE SDL3::SDL3 fastgltf::fastgltf glm::glm-header-only)
target_link_libraries(brutus-vulkan PRIVATE $<IF:$<TARGET_EXISTS:SDL3_image::SDL3_image-shared>,SDL3_image::SDL3_image-shared,SDL3_image::SDL3_image-static>)
target_link_libraries(brutus-vulkan PRIVATE renderer gpu_stable_power TBB::tbb imgui::imgui meshoptimizer::meshoptimizer mikktspace::mikktspace)

if(OPTICK_FOUND)
	target_link_libraries(brutus-vulkan PRIVATE Optick)
endif()