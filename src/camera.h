#pragma once

#include <glm/glm.hpp>

struct Camera
{
	glm::vec3 position;
	float pitch = 0.f;
	float yaw = 0.f;

	glm::mat4 calc_view_matrix() const;
	glm::mat4 calc_rotation_matrix() const;
};
