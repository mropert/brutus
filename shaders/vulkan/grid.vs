#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec2 outGridPos;
layout (location = 1) out vec3 outViewPos;

#include "constants.glsl"

layout( push_constant ) uniform constants
{	
	SceneConstants scene_constants;
} PushConstants;

const vec2 vertices[4] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
const float far_distance = 20;

void main() 
{	
    vec2 flat_pos = ( vertices[gl_VertexIndex] * far_distance ) + vec2( PushConstants.scene_constants.inverse_view_matrix[3][0], PushConstants.scene_constants.inverse_view_matrix[3][2] );
    vec3 world_pos = vec3( flat_pos.x, 0, flat_pos.y );

	gl_Position = PushConstants.scene_constants.view_projection_matrix * vec4(world_pos, 1);
    outGridPos = flat_pos + 0.5;
    outViewPos = ( PushConstants.scene_constants.view_matrix * vec4(world_pos, 1) ).xyz;
}
