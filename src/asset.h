#pragma once

#include <array>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/type_aligned.hpp>
#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/device.h>
#include <vector>

static constexpr size_t MESH_LOD_COUNT = 6;
static constexpr uint32_t TASK_GROUP_SIZE = 32; // Meshlets per task shader workgroup

int calc_mip_levels( renderer::Extent2D extent );

enum class material_type : uint32_t
{
	OPAQUE_SINGLE = 0,
	OPAQUE_DOUBLE,
	ALPHA_SINGLE,
	ALPHA_DOUBLE,
	COUNT
};

struct gpu_material
{
	glm::aligned_vec4 base_color;
	float metalness;
	float roughness;
	renderer::BindlessSampledTexture albedo_texture;
	renderer::BindlessSampledTexture normal_texture;
	renderer::BindlessSampledTexture metallic_roughness_texture;
	renderer::BindlessSampledTexture specular_texture;
	material_type type;
};

struct gpu_mesh_lod
{
	uint32_t indices_offset;
	uint32_t indices_count;
	uint32_t meshlets_offset;
	uint32_t meshlets_count;
	float error;
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
	std::array<gpu_mesh_lod, MESH_LOD_COUNT> lods;
	std::array<gpu_mesh_lod, MESH_LOD_COUNT> shadow_lods; // For depth-only rendering, only geometric error is considered
	renderer::BufferHandle vertices;
	renderer::BufferHandle meshlets;
	renderer::BufferHandle meshlet_vertices;
	renderer::BufferHandle meshlet_triangles;
};

struct index_buffer
{
	index_buffer( renderer::Device& device, size_t max_size )
		: buffer( device.create_buffer( renderer::Buffer::Usage::INDEX_BUFFER | renderer::Buffer::Usage::TRANSFER_DST,
										max_size * sizeof( uint32_t ) ) )
		, capacity( max_size )
	{
	}

	renderer::raii::Buffer buffer;
	std::atomic<size_t> size = 0;
	size_t capacity;
};

using BindlessManager = renderer::BindlessManager<gpu_mesh, gpu_material>;
using mesh_handle = renderer::BindlessHandle<gpu_mesh>;
using material_handle = renderer::BindlessHandle<gpu_material>;

struct mesh
{
	renderer::raii::Buffer data_buffer; // vertex + meshlets
	mesh_handle handle;
	size_t material_index;
	uint32_t meshlets_count; // All main LODs, excluding the shadow LODs
	uint32_t max_task_groups; // Task workgroups needed by its largest LOD, main or shadow
	// Data below is only for debugging purposes, a CPU side copy of the GPU-side descriptor
	gpu_mesh gpu_data;
};

struct material
{
	material_handle handle;
	material_type type;
	// Data below is only for debugging purposes, a CPU side copy of the GPU-side descriptor
	gpu_material gpu_data;
};

struct scene_node
{
	std::vector<uint32_t> children; // Use small vector once we get one
	glm::mat4 transform;
	size_t mesh = static_cast<size_t>( -1 );
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
