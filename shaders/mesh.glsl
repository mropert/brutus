struct Vertex
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 tangent;
};

struct Material
{
	vec4 baseColor;
	float metalness;
	float roughness;
	uint albedo_index;
	uint normal_index;
	uint metallic_roughness_index;
	uint specular_index;
	uint type;
	uint workflow;
	vec4 specular_glossiness;
	uint specular_glossiness_index;
	float alpha_cutoff;
};

// material_type in asset.h: pairs of single then double sided
bool is_single_sided( uint material_type )
{
	return ( material_type & 1 ) == 0;
}

const uint MATERIAL_WORKFLOW_METALLIC_ROUGHNESS = 0;
const uint MATERIAL_WORKFLOW_SPECULAR_GLOSSINESS = 1;

struct MeshLOD
{
	uint indices_offset;
	uint indices_count;
	uint meshlets_offset;
	uint meshlets_count;
	float error;
};

struct Meshlet
{
	uint vertex_offset;
	uint triangle_offset;
	uint vertex_count;
	uint triangle_count;
	vec4 bounding_sphere;
	vec4 cone;
};

struct MeshDraw
{
	vec3 translation;
	uint mesh;
	vec4 rotation;
	vec3 scale;
	uint material;
	uint meshlet_visibility_offset; // First bit of this draw in the meshlet visibility buffer, one bit per main LOD meshlet
};

layout( buffer_reference, std430 ) readonly buffer VertexBuffer
{
	Vertex vertices[];
};
layout( buffer_reference, std430 ) readonly buffer MaterialBuffer
{
	Material materials[];
};
layout( buffer_reference, std430 ) readonly buffer MeshletBuffer
{
	Meshlet meshlets[];
};
layout( buffer_reference, std430 ) readonly buffer MeshletVertexBuffer
{
	uint meshlet_vertices[];
};
layout( buffer_reference ) readonly buffer MeshletTriangleBuffer
{
	uint8_t meshlet_triangles[];
};
layout( buffer_reference, std430 ) readonly buffer MeshDrawBuffer
{
	MeshDraw mesh_draws[];
};

struct Mesh
{
	vec4 bounding_sphere;
	MeshLOD lods[ MESH_LOD_COUNT ];
	MeshLOD shadow_lods[ MESH_LOD_COUNT ]; // For depth-only rendering, only geometric error is considered
	VertexBuffer vb;
	MeshletBuffer mb;
	MeshletVertexBuffer mvb;
	MeshletTriangleBuffer mtb;
};

struct MeshletTaskPayload
{
	uint meshlets[ TASK_GROUP_SIZE ];
	uint draw_id;
};

// Vertex shader path, one per draw
struct MeshDrawCommand
{
	uint draw_id;
	uint lod;
	// VkDrawIndexedIndirectCommand
	uint index_count;
	uint instance_count;
	uint first_index;
	int vertex_offset;
	uint first_instance;
};

// Mesh shader path, one per task workgroup: all task workgroups of a bin are a single dispatch. The LOD is the bin's
struct MeshTaskCommand
{
	uint draw_id;
	// Relative to the LOD, the workgroup handles up to TASK_GROUP_SIZE meshlets from there. Late pass only, the top bit
	// is set when the early pass already drew the meshlets that were visible last frame
	uint first_meshlet;
};
const uint TASK_COMMAND_DRAWN_EARLY = 0x80000000;
