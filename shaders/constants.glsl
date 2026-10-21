layout( buffer_reference, std430 ) readonly buffer SceneConstants
{
	mat4 view_projection_matrix;
	mat4 projection_matrix;
	mat4 view_matrix;
	mat4 inverse_view_matrix;
	mat4 shadow_view_projection_matrix;
	vec4 frustum[ 6 ];
	vec3 camera_position;
	uint pbr;
	vec4 main_light;
	float sky_to_sun;
	uint shadow_map;
	uint shadow_pass; // Non-zero when rendering a shadow map: use shadow (depth-only) LODs
	vec2 viewport_size; // In pixels
	float shadow_distance; // View depth up to which the main light casts shadows
};
