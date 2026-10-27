#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_GOOGLE_include_directive : require

layout( location = 0 ) in vec2 inGridPos;
layout( location = 1 ) in vec3 inViewPos;
layout( location = 2 ) in vec3 inWorldPos;
layout( location = 0 ) out vec4 outFragColor;

#include "constants.glsl"

layout( set = 0, binding = 0 ) uniform texture2D textures[ 1024 ];
layout( set = 0, binding = 2 ) uniform sampler samplers[ 8 ];

layout( push_constant ) uniform constants
{
	vec3 grid_color;
	FrameConstants frame_constants;
	PassConstants pass_constants;
	float near_distance;
	float far_distance;
}
PushConstants;

#include "shadow.glsl"

const int GridMinorsPerMajor = 5;
const vec3 KeyColor[ 2 ] = { { 1, 0, 0 }, { 0, 0, 1 } };

// Inspired by https://vec3.ca/posts/rendering-an-infinite-grid

float Fade( float value )
{
	return 1 - smoothstep( PushConstants.near_distance, PushConstants.far_distance, value );
}

float ScaledFade( float value, vec2 fogScale )
{
	return 1 - smoothstep( PushConstants.near_distance * fogScale.x, PushConstants.far_distance * fogScale.y, value );
}

void main()
{
	const float viewDist = length( inViewPos );
	const float majorFog = Fade( viewDist );
	const float minorFog = ScaledFade( viewDist, vec2( 0.5, 0.85 ) );

	const ivec2 lineIndex = ivec2( floor( inGridPos ) );

	vec2 lineWidth;
	vec3 lineColor[ 2 ];
	vec2 lineFog;
	for ( int i = 0; i < 2; i++ )
	{
		float width;
		vec3 color;
		float fog;

		if ( lineIndex[ i ] == 0 )
		{
			width = 0.05;
			color = KeyColor[ 1 - i ];
			fog = majorFog;
		}
		else if ( abs( lineIndex[ i ] ) % GridMinorsPerMajor == 0 )
		{
			width = 0.0375;
			color = vec3( 0.5 );
			fog = majorFog;
		}
		else
		{
			width = 0.025;
			color = vec3( 0.25 );
			fog = minorFog;
		}

		lineWidth[ i ] = width;
		lineColor[ i ] = color;
		lineFog[ i ] = fog;
	}


	const vec2 lineDist = abs( 0.5 - fract( inGridPos ) );
	const vec2 lineAA = fwidth( inGridPos ) * 1.5;
	const vec2 lineMask = 1 - smoothstep( lineWidth - lineAA, lineWidth + lineAA, lineDist );
	vec2 blendFactors = lineMask * lineFog;

	for ( int i = 0; i < 2; i++ )
		if ( lineIndex[ 1 - i ] == 0 && lineIndex[ i ] != 0 )
			blendFactors[ i ] *= 1 - blendFactors[ 1 - i ];

	const float alpha = max( blendFactors.x, blendFactors.y );
	const vec3 color = max( lineColor[ 0 ] * blendFactors.x, lineColor[ 1 ] * blendFactors.y );
	// The grid is unlit, but receives the main light shadow: scale it the same way direct + ambient lighting would,
	// so unshadowed parts keep their color and shadowed ones only get the ambient part
	const vec3 up = vec3( 0, 1, 0 );
	const vec3 to_light = normalize( -PushConstants.frame_constants.main_light.xyz );
	const float NdotL = max( dot( up, to_light ), 0.0 );
	const float shadow = compute_shadow( PushConstants.frame_constants, inWorldPos, up, to_light );
	const float ambient = PushConstants.frame_constants.sky_to_sun;
	const float lighting = ( NdotL * shadow + ambient ) / ( NdotL + ambient );

	const vec3 tint = get_shadow_cascade_tint( PushConstants.frame_constants, inWorldPos );
	outFragColor = vec4( mix( PushConstants.grid_color, color, alpha ) * lighting * tint, 1 );
}
