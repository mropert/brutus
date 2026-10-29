#pragma once

#include <array>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_aligned.hpp>
#include <renderer/bindless.h>
#include <renderer/buffer.h>
#include <renderer/device.h>
#include <span>
#include <vector>

static constexpr size_t MESH_LOD_COUNT = 8;
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
	material_handle material;
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

struct scene_instance
{
	glm::vec3 translation;
	glm::quat rotation;
	glm::vec3 scale;
	mesh_handle mesh;
};

// A file's mesh nodes with their world transforms, the node hierarchy is flattened at load
struct scene
{
	std::vector<scene_instance> instances;
};

class Assets
{
public:
	Assets( renderer::Device& device, BindlessManager& bindless_manager, size_t max_indices );

	scene load( const std::filesystem::path& path );

	const material& get_material( material_handle h ) const { return _materials[ h.index ]; }
	const mesh& get_mesh( mesh_handle h ) const { return _meshes[ h.index ]; }
	std::span<const material> get_materials() const { return _materials; }
	std::span<const mesh> get_meshes() const { return _meshes; }

	const index_buffer& get_indices() const { return _indices; }

private:
	struct loaded_asset;
	class load_job;
	void add_content( loaded_asset& asset );

	renderer::Device& _device;
	BindlessManager& _bindless_manager;
	std::vector<material> _materials;
	std::vector<mesh> _meshes;
	index_buffer _indices;
};
