#version 460

void main()
{
	// Generate coordinates for a single large triangle covering [-1, 1]
	const vec2 uv = vec2( gl_VertexIndex & 2, ( gl_VertexIndex << 1 ) & 2 );
	gl_Position = vec4( uv * vec2( 2.0, -2.0 ) + vec2( -1.0, 1.0 ), 0.0, 1.0 );
}
