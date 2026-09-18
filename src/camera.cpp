#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

glm::mat4 Camera::calc_view_matrix() const
{
	const glm::mat4 camera_translation = glm::translate( glm::mat4( 1.f ), position );
	const glm::mat4 camera_rotation = calc_rotation_matrix();
	return glm::inverse( camera_translation * camera_rotation );
}

glm::mat4 Camera::calc_rotation_matrix() const
{
	glm::quat pitch_rotation = glm::angleAxis( pitch, glm::vec3 { 1.f, 0.f, 0.f } );
	glm::quat yaw_rotation = glm::angleAxis( yaw, glm::vec3 { 0.f, -1.f, 0.f } );

	return glm::mat4_cast( yaw_rotation ) * glm::mat4_cast( pitch_rotation );
}
