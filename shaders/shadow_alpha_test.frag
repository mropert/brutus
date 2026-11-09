#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_GOOGLE_include_directive : require

#include "constants.glsl"
#include "mesh.glsl"

// Depth only, the alpha test of mesh.frag
layout( location = 0, component = 3 ) in float inU;
layout( location = 1, component = 3 ) in float inV;
layout( location = 3 ) in flat uint inPackedMaterial;

layout( set = 0, binding = 0 ) uniform texture2D textures[ 1024 ];
layout( set = 0, binding = 2 ) uniform sampler samplers[ 8 ];
layout( std430, set = 1, binding = 1 ) readonly buffer ssbo2
{
	Material materials[];
};

void main()
{
	const vec2 uv = vec2( inU, inV );
	const uint material = unpack_material( inPackedMaterial );
	float alpha = materials[ material ].base_color.a;
	if ( materials[ material ].albedo_index != -1 )
	{
		alpha *= texture( sampler2D( textures[ materials[ material ].albedo_index ], samplers[ 0 ] ), uv ).a;
	}

	if ( alpha < materials[ material ].alpha_cutoff )
	{
		discard;
	}
}
