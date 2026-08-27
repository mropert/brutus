#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D textures[1024];

layout( push_constant, std430 ) uniform block
{
    uint tex_index;
};

void main() {
    vec3 color = texelFetch(textures[tex_index], ivec2(gl_FragCoord.xy), 0).rgb;
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
