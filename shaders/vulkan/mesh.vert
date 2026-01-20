#version 450
#extension GL_ARB_shader_draw_parameters : require
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
layout (location = 5) out flat uint outMaterial;

layout(std430, set = 1, binding = 0) readonly buffer ssbo1 { Mesh meshes[]; };

layout(buffer_reference, std430) readonly buffer DrawCommandsBuffer { MeshDrawCommand commands[]; };


layout( push_constant, std430 ) uniform block
{
    SceneConstants scene_constants;
    MeshDrawBuffer draws_buffer;
    DrawCommandsBuffer commands_buffer;
};

void main() 
{   
    const uint draw_id = commands_buffer.commands[gl_DrawIDARB].draw_id;
    const uint mesh_id = draws_buffer.mesh_draws[draw_id].mesh;

    const Vertex v = meshes[mesh_id].vb.vertices[gl_VertexIndex];

    const vec3 pos =  draws_buffer.mesh_draws[draw_id].position + v.position;
    gl_Position = scene_constants.view_projection_matrix * vec4( pos, 1 );
    outUV.x = v.uv_x;
    outUV.y = v.uv_y;
    outNormal = v.normal;
    outWorldPos = pos;
    outTangent = v.tangent.xyz;
    outBitangent = cross(outNormal, outTangent) * v.tangent.w;
    outMaterial = draws_buffer.mesh_draws[draw_id].material;
}
