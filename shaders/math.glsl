// GLSL doesn't provide this constant, unlike C. Since GLSL 4.40 (section 4.7.1), dividing a non-zero value by zero gives
// an infinity: glslang folds this into the +infinity constant
const float INFINITY = 1.0 / 0.0;

vec3 rotate_by_quaternion( vec3 v, vec4 q )
{
	return v + 2.0 * cross( q.xyz, cross( q.xyz, v ) + q.w * v );
}

// 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere. Michael Mara, Morgan McGuire. 2013
bool projectSphere( vec3 c, float r, float znear, float P00, float P11, out vec4 aabb )
{
	if ( c.z < r + znear )
		return false;

	vec3 cr = c * r;
	float czr2 = c.z * c.z - r * r;

	float vx = sqrt( c.x * c.x + czr2 );
	float minx = ( vx * c.x - cr.z ) / ( vx * c.z + cr.x );
	float maxx = ( vx * c.x + cr.z ) / ( vx * c.z - cr.x );

	float vy = sqrt( c.y * c.y + czr2 );
	float miny = ( vy * c.y - cr.z ) / ( vy * c.z + cr.y );
	float maxy = ( vy * c.y + cr.z ) / ( vy * c.z - cr.y );

	aabb = vec4( minx * P00, miny * P11, maxx * P00, maxy * P11 );
	aabb = aabb.xwzy * vec4( 0.5f, -0.5f, 0.5f, -0.5f ) + vec4( 0.5f ); // clip space -> uv space

	return true;
}
