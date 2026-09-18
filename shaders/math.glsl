vec3 rotate_by_quaternion( vec3 v, vec4 q )
{
	return v + 2.0 * cross( q.xyz, cross( q.xyz, v ) + q.w * v );
}
