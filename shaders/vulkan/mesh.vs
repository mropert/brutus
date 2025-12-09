#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outWorldPos;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outTangent;
layout (location = 4) out vec3 outBitangent;

struct Vertex {
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 tangent;
}; 

#include "constants.glsl"

layout(buffer_reference, std430) readonly buffer VertexBuffer { 
	Vertex vertices[];
};

layout( push_constant ) uniform constants
{	
	layout(offset = 32) SceneConstants scene_constants;
	VertexBuffer vertex_buffer;
} PushConstants;

void main() 
{	
	Vertex v = PushConstants.vertex_buffer.vertices[gl_VertexIndex];

	gl_Position = PushConstants.scene_constants.view_projection_matrix * vec4(v.position, 1.0f);
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
    outNormal = v.normal;
    outWorldPos = v.position;
    outTangent = v.tangent.xyz;
    outBitangent = cross(outNormal, outTangent) * v.tangent.w;
}
