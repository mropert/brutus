#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_GOOGLE_include_directive : require

#include "constants.glsl"
#include "mesh.glsl"

// Depth only, the alpha test of mesh.frag
layout( location = 0 ) in vec2 inUV;
layout( location = 4 ) in flat uint inMaterial;

layout( set = 0, binding = 0 ) uniform texture2D textures[ 1024 ];
layout( set = 0, binding = 2 ) uniform sampler samplers[ 8 ];
layout( std430, set = 1, binding = 1 ) readonly buffer ssbo2
{
	Material materials[];
};

void main()
{
	const uint albedo_index = materials[ inMaterial ].albedo_index;
	if ( albedo_index == -1 )
	{
		return;
	}
	const float alpha = texture( sampler2D( textures[ albedo_index ], samplers[ 0 ] ), inUV ).a * materials[ inMaterial ].baseColor.a;
	if ( alpha < materials[ inMaterial ].alpha_cutoff )
	{
		discard;
	}
}
