#pragma once

#include <array>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/type_aligned.hpp>
#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/device.h>
#include <vector>

struct pbr_data
{
	glm::vec4 base_color;
	float metalness;
	float roughness;
};

struct gpu_material
{
	pbr_data pbr_data;
	uint32_t albedo_texture_index;
	uint32_t normal_texture_index = -1;
	uint32_t metallic_roughness_texture_index = -1;
	uint32_t padding[ 3 ];
};

struct gpu_mesh_lod
{
	uint32_t indices_offset;
	uint32_t indices_count;
	uint32_t meshlets_offset;
	uint32_t meshlets_count;
};

struct gpu_meshlet
{
	uint32_t vertex_offset;
	uint32_t triangle_offset;
	uint32_t vertex_count;
	uint32_t triangle_count;
	glm::aligned_vec4 bounding_sphere; // w is radius
	glm::aligned_vec4 cone; // xyz is axis, w is cutoff
};

struct gpu_mesh
{
	glm::aligned_vec4 bounding_sphere; // z is radius
	std::array<gpu_mesh_lod, 4> lods;
	renderer::BufferHandle vertices;
	renderer::BufferHandle meshlets;
	renderer::BufferHandle meshlet_vertices;
	renderer::BufferHandle meshlet_triangles;
};

struct index_buffer
{
	index_buffer( renderer::Device& device, uint32_t max_size )
		: buffer( device.create_buffer( renderer::Buffer::Usage::INDEX_BUFFER | renderer::Buffer::Usage::TRANSFER_DST,
										max_size * sizeof( uint32_t ) ) )
		, capacity( max_size )
	{
	}

	renderer::raii::Buffer buffer;
	std::atomic<uint32_t> size = 0;
	uint32_t capacity;
};

using BindlessManager = renderer::BindlessManager<gpu_mesh, gpu_material>;
using mesh_handle = renderer::BindlessHandle<gpu_mesh>;
using material_handle = renderer::BindlessHandle<gpu_material>;

struct mesh
{
	renderer::raii::Buffer data_buffer; // vertex + meshlets
	mesh_handle handle;
	uint32_t material_index;
	// Data below is only for debugging purposes, a CPU side copy of the GPU-side descriptor
	gpu_mesh gpu_data;
};

struct material
{
	material_handle handle;
	// Data below is only for debugging purposes, a CPU side copy of the GPU-side descriptor
	gpu_material gpu_data;
};

struct scene_node
{
	std::vector<uint32_t> children;	// Use small vector once we get one
	glm::mat4 transform;
	uint32_t mesh = -1;
};

struct asset
{
	std::vector<mesh> meshes;
	std::vector<material> materials;
	glm::vec4 bounding_sphere;
	std::vector<scene_node> nodes;
	std::vector<uint32_t> scene_roots;
};

asset load_asset( renderer::Device& device, BindlessManager& bindless_manager, index_buffer& index_buffer, std::filesystem::path path );
