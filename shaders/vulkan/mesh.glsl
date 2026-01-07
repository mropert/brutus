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

struct Mesh
{
    Material material;
    MeshLOD lods[4];
    uint vertices_offset;
};

struct Meshlet {
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
};

struct MeshletTaskPayload
{
    vec3 position;
    uint lod;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer { Vertex vertices[]; };
layout(buffer_reference, std430) readonly buffer InstanceBuffer { mat4 model_matrix[]; };
layout(buffer_reference, std430) readonly buffer MaterialBuffer { Material materials[]; };
layout(buffer_reference, std430) readonly buffer MeshletBuffer { Meshlet meshlets[]; };
layout(buffer_reference, std430) readonly buffer MeshletVertexBuffer { uint meshlet_vertices[]; };
layout(buffer_reference) readonly buffer MeshletTriangleBuffer { uint8_t meshlet_triangles[]; };