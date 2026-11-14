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
	vec4 base_color;
	vec3 emissive;
	vec4 specular_glossiness;
	uint albedo_index;
	uint normal_index;
	uint metallic_roughness_index;
	uint occlusion_index;
	uint emissive_index;
	uint specular_index;
	uint specular_glossiness_index;
	float metallic;
	float roughness;
	float alpha_cutoff;
	float occlusion_strength;
	uint type;
	uint workflow;
};

// The material index goes to the fragment shaders as a flat uint, with the bitangent sign of the tangent in the top bit.
// It's flat (the provoking vertex's), which only differs from interpolating it on triangles mixing both signs
uint pack_material( uint material, float bitangent_sign )
{
	return material | ( bitangent_sign < 0.0 ? 0x80000000u : 0u );
}

uint unpack_material( uint packed_material )
{
	return packed_material & 0x7fffffffu;
}

float unpack_bitangent_sign( uint packed_material )
{
	return ( packed_material & 0x80000000u ) != 0u ? -1.0 : 1.0;
}

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
	vec3 shadow_lod_offset; // See shadow_lod_offset()
};

// Shadow passes: simplified LODs deviate from the full detail surface by up to their error, so where they rise above it
// they shadow the receivers drawn at full detail in the main view. Moving them away from the light by their error keeps
// them under it. Zero for LOD 0 and outside shadow passes, lod is a shadow LOD
vec3 shadow_lod_offset( PassConstants constants, MeshLOD lod, vec3 scale )
{
	if ( constants.data.shadow_pass == 0 )
	{
		return vec3( 0.0 );
	}
	return constants.data.shadow_caster_sweep.xyz * ( lod.error * max( scale.x, max( scale.y, scale.z ) ) );
}

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
