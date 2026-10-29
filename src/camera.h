#pragma once

#include <glm/glm.hpp>

struct Camera
{
	// Vertical, in radians: 50.70 degrees. It used to be 70 degrees passed as radians by accident, which gave this angle;
	// kept so that reference images and benchmarks stay comparable
	static constexpr float DEFAULT_FOV = 0.884962f;

	glm::vec3 position;
	float pitch = 0.f;
	float yaw = 0.f;
	float fov = DEFAULT_FOV; // Vertical, in radians

	glm::mat4 calc_view_matrix() const;
	glm::mat4 calc_rotation_matrix() const;
};
