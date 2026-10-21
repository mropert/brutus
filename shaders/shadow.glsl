// Main light shadow lookup
// Requires constants.glsl and the bindless textures/samplers declarations

// Fraction of the shadow distance over which shadows fade out
const float SHADOW_FADE_RANGE = 0.1;

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

	// Fade out based on view depth, matching how the shadow map bounds are fitted to the view frustum
	const vec3 camera_forward = -constants.inverse_view_matrix[ 2 ].xyz;
	const float view_depth = dot( world_pos - constants.camera_position, camera_forward );
	const float fade = 1.0 - smoothstep( ( 1.0 - SHADOW_FADE_RANGE ) * constants.shadow_distance, constants.shadow_distance, view_depth );
	if ( fade <= 0.0 )
	{
		return 1.0;
	}

	const vec4 shadow_pos = constants.shadow_view_projection_matrix * vec4( world_pos, 1.0 );
	const vec3 shadow_ndc = shadow_pos.xyz / shadow_pos.w;
	const vec2 shadow_uv = vec2( shadow_ndc.x * 0.5 + 0.5, 0.5 - shadow_ndc.y * 0.5 );
	if ( any( lessThan( shadow_uv, vec2( 0.0 ) ) ) || any( greaterThan( shadow_uv, vec2( 1.0 ) ) ) || shadow_ndc.z < 0.0
		 || shadow_ndc.z > 1.0 )
	{
		return 1.0;
	}

	const float shadow_bias = max( 0.0005 * ( 1.0 - NdotL ), 0.0005 );
	const float shadow = texture( sampler2DShadow( textures[ constants.shadow_map ], samplers[ 2 ] ),
								  vec3( shadow_uv, shadow_ndc.z - shadow_bias ) );
	return mix( 1.0, shadow, fade );
}
