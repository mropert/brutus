#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_GOOGLE_include_directive : require

layout( location = 0 ) out vec2 outGridPos;
layout( location = 1 ) out vec3 outViewPos;
layout( location = 2 ) out vec3 outWorldPos;

#include "constants.glsl"

layout( push_constant ) uniform constants
{
	vec3 grid_color;
	FrameConstants frame_constants;
	PassConstants pass_constants;
	float near_distance;
	float far_distance;
}
PushConstants;

const vec2 vertices[ 4 ] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };

void main()
{
	const vec2 flat_pos = ( vertices[ gl_VertexIndex ] * PushConstants.far_distance )
		+ vec2( PushConstants.pass_constants.data.inverse_view_matrix[ 3 ][ 0 ],
				PushConstants.pass_constants.data.inverse_view_matrix[ 3 ][ 2 ] );
	const vec3 world_pos = vec3( flat_pos.x, 0, flat_pos.y );

	gl_Position = PushConstants.pass_constants.data.view_projection_matrix * vec4( world_pos, 1 );
	outGridPos = flat_pos + 0.5;
	outViewPos = ( PushConstants.pass_constants.data.view_matrix * vec4( world_pos, 1 ) ).xyz;
	outWorldPos = world_pos;
}
