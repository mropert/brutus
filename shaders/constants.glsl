const int SHADOW_CASCADE_MAX = 4;

layout( buffer_reference, std430 ) readonly buffer SceneConstants
{
	mat4 view_projection_matrix;
	mat4 projection_matrix;
	mat4 view_matrix;
	mat4 inverse_view_matrix;
	mat4 shadow_view_projection_matrices[ SHADOW_CASCADE_MAX ];
	vec4 shadow_cascade_splits; // Far view depth of each cascade
	vec4 shadow_cascade_texel_sizes; // World size of a shadow map texel in each cascade
	vec4 frustum[ 6 ];
	// Shadow passes only: the receivers that sample this cascade (camera frustum sides and the view depth range of the
	// cascade, including the previous cascade's blend band) and the caster sweep, xyz the light direction, w the farthest
	// receiver along it
	vec4 shadow_receiver_planes[ 6 ];
	vec4 shadow_caster_sweep;
	vec3 camera_position;
	uint pbr;
	vec4 main_light;
	float sky_to_sun;
	uint shadow_map;
	uint shadow_pass; // Non-zero when rendering a shadow map: use shadow (depth-only) LODs
	vec2 viewport_size; // In pixels
	float shadow_distance; // View depth up to which the main light casts shadows
	uint shadow_cascade_count;
	uint show_shadow_cascades;
	uint sample_count; // Of the render target, must use the standard sample locations
};

// Shadow passes: false when the shadow of the sphere can't reach any receiver of the cascade. The sphere is swept along
// the light up to the farthest receiver, and the resulting capsule is culled when it's outside one of the receiver planes
bool shadow_caster_visible( SceneConstants constants, vec3 center, float radius )
{
	const vec3 light_direction = constants.shadow_caster_sweep.xyz;
	const float sweep_length = constants.shadow_caster_sweep.w - dot( center, light_direction );
	const vec3 sweep_end = center + light_direction * max( sweep_length, 0.0 );
	bool visible = sweep_length > -radius;
	for ( int i = 0; i < 6; ++i )
	{
		const vec4 plane = constants.shadow_receiver_planes[ i ];
		visible = visible && max( dot( plane, vec4( center, 1 ) ), dot( plane, vec4( sweep_end, 1 ) ) ) > -radius;
	}
	return visible;
}
