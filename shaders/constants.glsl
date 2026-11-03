const int SHADOW_CASCADE_MAX = 4;
// Fraction of each cascade's depth slice over which it blends into the next one
const float SHADOW_CASCADE_BLEND_RANGE = 0.25;

// Constants written on the GPU are structs: access qualifiers of buffer references belong to the block type, not to the
// variables, so readers and writers each get their own block around the struct

// Fitted to the visible depth range by shadow_cascades.comp each frame
struct ShadowCascadesData
{
	mat4 view_projection_matrices[ SHADOW_CASCADE_MAX ];
	vec4 splits; // Far view depth of each cascade
	vec4 texel_sizes; // World size of a shadow map texel in each cascade
};
layout( buffer_reference, std430 ) readonly buffer ShadowCascades
{
	ShadowCascadesData data;
};

// Written once per frame, shared by every pass
layout( buffer_reference, std430 ) readonly buffer FrameConstants
{
	// Main camera
	vec3 camera_position;
	vec3 camera_forward;

	// Lighting
	vec4 main_light; // xyz the direction the light travels, w its intensity
	float sky_to_sun;
	uint pbr;

	// Main light shadows
	uint shadow_map; // -1 when shadows are off
	uint shadow_cascade_count;
	float shadow_distance; // View depth up to which the main light casts shadows, the cascades may stop before
	ShadowCascades shadow_cascades;

	// Debug
	uint show_shadow_cascades;
};

// The view and render target of one pass: the main view (written by the CPU) or a shadow cascade (shadow_cascades.comp)
struct PassConstantsData
{
	// View
	mat4 view_projection_matrix;
	mat4 projection_matrix; // [3][3] is 0 for perspective projections, 1 for orthographic ones
	mat4 view_matrix;
	mat4 inverse_view_matrix;
	vec3 view_position;
	vec4 frustum[ 6 ];

	// Render target
	vec2 viewport_size; // In pixels
	uint sample_count; // Must use the standard sample locations

	// Culling selects the coarsest LOD whose error is under lod_target, scaled by distance for perspective projections
	float lod_target;

	// Shadow passes only
	uint shadow_pass; // Non-zero when rendering a shadow map: use shadow (depth-only) LODs
	// The receivers that sample this cascade (camera frustum sides and the view depth range of the cascade, including the
	// previous cascade's blend band) and the caster sweep, xyz the light direction, w the farthest receiver along it
	vec4 shadow_receiver_planes[ 6 ];
	vec4 shadow_caster_sweep;
};
layout( buffer_reference, std430 ) readonly buffer PassConstants
{
	PassConstantsData data;
};

// Shadow passes: false when the shadow of the sphere can't reach any receiver of the cascade. The sphere is swept along
// the light up to the farthest receiver, and the resulting capsule is culled when it's outside one of the receiver planes
bool shadow_caster_visible( PassConstants constants, vec3 center, float radius )
{
	const vec3 light_direction = constants.data.shadow_caster_sweep.xyz;
	const float sweep_length = constants.data.shadow_caster_sweep.w - dot( center, light_direction );
	const vec3 sweep_end = center + light_direction * max( sweep_length, 0.0 );
	bool visible = sweep_length > -radius;
	for ( int i = 0; i < 6; ++i )
	{
		const vec4 plane = constants.data.shadow_receiver_planes[ i ];
		visible = visible && max( dot( plane, vec4( center, 1 ) ), dot( plane, vec4( sweep_end, 1 ) ) ) > -radius;
	}
	return visible;
}
