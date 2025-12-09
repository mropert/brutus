#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) in vec2 inGridPos;
layout (location = 1) in vec3 inViewPos;
layout (location = 0) out vec4 outFragColor;

void main()
{
    outFragColor = vec4( 1, 0, 1, 1 );
}
