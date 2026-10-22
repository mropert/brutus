// Main light shadow lookup
// Requires constants.glsl and the bindless textures/samplers declarations

// Array textures (cascades) alias the bindless textures binding
layout( set = 0, binding = 0 ) uniform texture2DArray textures_2d_array[ 1024 ];

// Fraction of the shadow distance over which shadows fade out
const float SHADOW_FADE_RANGE = 0.1;
// Normal offset at grazing angles, in texels of the selected cascade
const float SHADOW_NORMAL_OFFSET_TEXELS = 1.5;
// Fraction of each cascade's depth slice over which it blends into the next one
const float SHADOW_CASCADE_BLEND_RANGE = 0.1;

// Shadow maps are fitted to the view frustum, so they are selected and faded based on view depth
float shadow_view_depth( SceneConstants constants, vec3 world_pos )
{
	const vec3 camera_forward = -constants.inverse_view_matrix[ 2 ].xyz;
	return dot( world_pos - constants.camera_position, camera_forward );
}

// Cascades are fitted to consecutive view depth slices
uint select_shadow_cascade( SceneConstants constants, float view_depth )
{
	uint cascade = 0;
	while ( cascade + 1 < constants.shadow_cascade_count && view_depth > constants.shadow_cascade_splits[ cascade ] )
	{
		++cascade;
	}
	return cascade;
}

// Color tint of the cascade covering world_pos for debugging, (1, 1, 1) when disabled or outside of cascades
vec3 get_shadow_cascade_tint( SceneConstants constants, vec3 world_pos )
{
	const float view_depth = shadow_view_depth( constants, world_pos );
	if ( constants.show_shadow_cascades == 0 || constants.shadow_map == -1 || view_depth > constants.shadow_distance )
	{
		return vec3( 1.0 );
	}
	const vec3 tints[ SHADOW_CASCADE_MAX ] = { vec3( 1.0, 0.3, 0.3 ), vec3( 0.3, 1.0, 0.3 ), vec3( 0.3, 0.3, 1.0 ), vec3( 1.0, 1.0, 0.3 ) };
	return tints[ select_shadow_cascade( constants, view_depth ) ];
}

// Returns 1 when lit, 0 when shadowed, -1 when world_pos is outside of the cascade
float sample_shadow_cascade( SceneConstants constants, vec3 world_pos, vec3 normal, float sin_angle, uint cascade )
{
	// Normal offset: move the lookup off the surface by a few texels, more at grazing angles where a texel covers
	// a larger depth range of the receiver
	const vec3 offset_pos = world_pos
		+ normal * ( constants.shadow_cascade_texel_sizes[ cascade ] * SHADOW_NORMAL_OFFSET_TEXELS * sin_angle );
	const vec4 shadow_pos = constants.shadow_view_projection_matrices[ cascade ] * vec4( offset_pos, 1.0 );
	const vec3 shadow_ndc = shadow_pos.xyz / shadow_pos.w;
	const vec2 shadow_uv = vec2( shadow_ndc.x * 0.5 + 0.5, 0.5 - shadow_ndc.y * 0.5 );
	if ( any( lessThan( shadow_uv, vec2( 0.0 ) ) ) || any( greaterThan( shadow_uv, vec2( 1.0 ) ) ) || shadow_ndc.z < 0.0
		 || shadow_ndc.z > 1.0 )
	{
		return -1.0;
	}

	return texture( sampler2DArrayShadow( textures_2d_array[ constants.shadow_map ], samplers[ 2 ] ),
					vec4( shadow_uv, cascade, shadow_ndc.z ) );
}

// Returns 1 when fully lit, 0 when fully shadowed
// normal is the geometric normal (facing the viewer for double sided surfaces), to_light the normalized direction towards the light
float compute_shadow( SceneConstants constants, vec3 world_pos, vec3 normal, vec3 to_light )
{
	if ( constants.shadow_map == -1 )
	{
		return 1.0;
	}

	// Surfaces facing away from the light are already unlit, no need to sample
	const float NdotL = dot( normal, to_light );
	if ( NdotL <= 0.0 )
	{
		return 1.0;
	}

	const float view_depth = shadow_view_depth( constants, world_pos );
	const float fade = 1.0 - smoothstep( ( 1.0 - SHADOW_FADE_RANGE ) * constants.shadow_distance, constants.shadow_distance, view_depth );
	if ( fade <= 0.0 )
	{
		return 1.0;
	}

	const uint cascade = select_shadow_cascade( constants, view_depth );
	const float sin_angle = sqrt( 1.0 - NdotL * NdotL );
	float shadow = sample_shadow_cascade( constants, world_pos, normal, sin_angle, cascade );
	if ( shadow < 0.0 )
	{
		return 1.0;
	}

	// Blend towards the next cascade over the end of this one so the resolution change isn't a hard edge. The last
	// cascade is covered by the distance fade instead
	if ( cascade + 1 < constants.shadow_cascade_count )
	{
		const float cascade_far = constants.shadow_cascade_splits[ cascade ];
		const float cascade_near = cascade == 0 ? 0.0 : constants.shadow_cascade_splits[ cascade - 1 ];
		const float blend_start = cascade_far - SHADOW_CASCADE_BLEND_RANGE * ( cascade_far - cascade_near );
		const float blend = smoothstep( blend_start, cascade_far, view_depth );
		if ( blend > 0.0 )
		{
			// The next cascade is fitted to its own depth slice and may not cover this point, keep this one then
			const float next_shadow = sample_shadow_cascade( constants, world_pos, normal, sin_angle, cascade + 1 );
			if ( next_shadow >= 0.0 )
			{
				shadow = mix( shadow, next_shadow, blend );
			}
		}
	}

	return mix( 1.0, shadow, fade );
}
