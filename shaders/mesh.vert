#version 450
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive : require

#include "constants.glsl"
#include "math.glsl"
#include "mesh.glsl"

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outWorldPos;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec4 outTangent;
layout (location = 4) out flat uint outMaterial;

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
    const uint draw_id = commands_buffer.commands[gl_BaseInstanceARB].draw_id;
    const uint mesh_id = draws_buffer.mesh_draws[draw_id].mesh;

    const Vertex v = meshes[mesh_id].vb.vertices[gl_VertexIndex];

    const vec3 pos = rotate_by_quaternion( v.position, draws_buffer.mesh_draws[draw_id].rotation ) 
                    * draws_buffer.mesh_draws[draw_id].scale
                    + draws_buffer.mesh_draws[draw_id].translation;
    gl_Position = scene_constants.view_projection_matrix * vec4( pos, 1 );
    outUV.x = v.uv_x;
    outUV.y = v.uv_y;
    outNormal = rotate_by_quaternion( v.normal, draws_buffer.mesh_draws[draw_id].rotation );
    outWorldPos = pos;
    outTangent = vec4( rotate_by_quaternion( v.tangent.xyz, draws_buffer.mesh_draws[draw_id].rotation ), v.tangent.w);
    outMaterial = draws_buffer.mesh_draws[draw_id].material;
}
