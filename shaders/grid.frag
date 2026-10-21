#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_GOOGLE_include_directive : require

layout( location = 0 ) in vec2 inGridPos;
layout( location = 1 ) in vec3 inViewPos;
layout( location = 0 ) out vec4 outFragColor;

#include "constants.glsl"

layout( push_constant ) uniform constants
{
	vec3 grid_color;
	SceneConstants scene_constants;
	float near_distance;
	float far_distance;
}
PushConstants;

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
	outFragColor = vec4( mix( PushConstants.grid_color, color, alpha ), 1 );
}
