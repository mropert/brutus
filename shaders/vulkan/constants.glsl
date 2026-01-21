layout(buffer_reference, std430) readonly buffer SceneConstants {
	mat4 view_projection_matrix;
    mat4 view_matrix;
	mat4 inverse_view_matrix;
	vec4 frustum[6];
	vec3 camera_position;
	uint pbr;
	vec4 main_light;
};
