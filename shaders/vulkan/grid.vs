#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec2 outGridPos;
layout (location = 1) out vec3 outViewPos;

layout(buffer_reference, std430) readonly buffer SceneConstants {
	mat4 view_projection_matrix;
	vec3 camera_position;
	vec4 main_light;
};

layout( push_constant ) uniform constants
{	
	SceneConstants scene_constants;
} PushConstants;

const vec2 vertices[4] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };

void main() 
{	
    vec3 worldPos = vec3( vertices[gl_VertexIndex].x, 0, vertices[gl_VertexIndex].y );

	gl_Position = PushConstants.scene_constants.view_projection_matrix * vec4(worldPos, 1.0);
}
