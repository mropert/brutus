#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive : require

#include "constants.glsl"
#include "mesh.glsl"

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outWorldPos;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outTangent;
layout (location = 4) out vec3 outBitangent;

layout(std430, set = 0, binding = 2) readonly buffer ssbo1 { Mesh meshes[]; };

layout( push_constant, std430 ) uniform constants
{
    uint mesh;
    uint material;
    SceneConstants scene_constants;
    InstanceBuffer instance_buffer;
} PushConstants;

void main() 
{
    const Vertex v = meshes[PushConstants.mesh].vb.vertices[gl_VertexIndex];

    const vec4 pos = PushConstants.instance_buffer.model_matrix[gl_InstanceIndex] * vec4(v.position, 1.0f);
    gl_Position = PushConstants.scene_constants.view_projection_matrix * pos;
    outUV.x = v.uv_x;
    outUV.y = v.uv_y;
    outNormal = v.normal;
    outWorldPos = pos.xyz;
    outTangent = v.tangent.xyz;
    outBitangent = cross(outNormal, outTangent) * v.tangent.w;
}
