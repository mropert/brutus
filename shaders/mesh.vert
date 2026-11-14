#version 460
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_GOOGLE_include_directive : require

#include "constants.glsl"
#include "math.glsl"
#include "mesh.glsl"

// Same interpolants as mesh.mesh
layout( location = 0 ) out vec3 outNormal;
layout( location = 0, component = 3 ) out float outU;
layout( location = 1 ) out vec3 outTangent;
layout( location = 1, component = 3 ) out float outV;
layout( location = 2 ) out vec3 outWorldPos;
layout( location = 3 ) out flat uint outMaterial;

layout( std430, set = 1, binding = 0 ) readonly buffer ssbo1
{
	Mesh meshes[];
};

layout( buffer_reference, std430 ) readonly buffer DrawCommandsBuffer
{
	MeshDrawCommand commands[];
};


layout( push_constant, std430 ) uniform block
{
	FrameConstants frame_constants;
	PassConstants pass_constants;
	MeshDrawBuffer draws_buffer;
	DrawCommandsBuffer commands_buffer;
};

void main()
{
	const uint draw_id = commands_buffer.commands[ gl_BaseInstanceARB ].draw_id;
	const uint mesh_id = draws_buffer.mesh_draws[ draw_id ].mesh;
	const MeshLOD shadow_lod = meshes[ mesh_id ].shadow_lods[ commands_buffer.commands[ gl_BaseInstanceARB ].lod ];

	const Vertex v = meshes[ mesh_id ].vb.vertices[ gl_VertexIndex ];

	const vec3 pos = rotate_by_quaternion( v.position, draws_buffer.mesh_draws[ draw_id ].rotation )
			* draws_buffer.mesh_draws[ draw_id ].scale
		+ draws_buffer.mesh_draws[ draw_id ].translation
		+ shadow_lod_offset( pass_constants, shadow_lod, draws_buffer.mesh_draws[ draw_id ].scale );
	gl_Position = pass_constants.data.view_projection_matrix * vec4( pos, 1 );
	outNormal = rotate_by_quaternion( v.normal, draws_buffer.mesh_draws[ draw_id ].rotation );
	outU = v.uv_x;
	outTangent = rotate_by_quaternion( v.tangent.xyz, draws_buffer.mesh_draws[ draw_id ].rotation );
	outV = v.uv_y;
	outWorldPos = pos;
	outMaterial = pack_material( draws_buffer.mesh_draws[ draw_id ].material, v.tangent.w );
}
