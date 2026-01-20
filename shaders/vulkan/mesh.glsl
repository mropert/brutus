struct Vertex {
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
};

struct MeshLOD
{
    uint indices_offset;
    uint indices_count;
    uint meshlets_offset;
    uint meshlets_count;
};

struct Meshlet {
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
    vec4 bounding_sphere;
};

struct MeshDraw
{
    vec3 position;
    float padding;
    uint mesh;
    uint material;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer { Vertex vertices[]; };
layout(buffer_reference, std430) readonly buffer MaterialBuffer { Material materials[]; };
layout(buffer_reference, std430) readonly buffer MeshletBuffer { Meshlet meshlets[]; };
layout(buffer_reference, std430) readonly buffer MeshletVertexBuffer { uint meshlet_vertices[]; };
layout(buffer_reference) readonly buffer MeshletTriangleBuffer { uint8_t meshlet_triangles[]; };
layout(buffer_reference, std430) readonly buffer MeshDrawBuffer { MeshDraw mesh_draws[]; };

struct Mesh
{
    vec4 bounding_sphere;
    MeshLOD lods[4];
    VertexBuffer vb;
    MeshletBuffer mb;
    MeshletVertexBuffer mvb;
    MeshletTriangleBuffer mtb;
};

struct MeshletTaskPayload
{
    vec3 position;
    uint lod;
};

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
    // VkDrawMeshTasksIndirectCommandEXT
    uint group_count_x;
    uint group_count_y;
    uint group_count_z;
};
