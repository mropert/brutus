#version 450

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inFragPos;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec4 inTangent;
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D albedoTexture;
layout(set = 0, binding = 1) uniform sampler2D normalTexture;

const vec3 lightPos = vec3(-10.0, 0, -10.0);

void main() 
{
	float ambient = 0.1;
	vec3 normal = normalize(inNormal);
	vec3 tangent = normalize(inTangent.xyz);
	vec3 normalDisplacement = normalize((texture(normalTexture, inUV).xyz * 2.0) - 1.0);
	normalDisplacement.y *= inTangent.w;
	mat3 TBN = mat3(tangent, cross(normal, tangent), normal	);
	vec3 combinedNormal = normalize(TBN * normalDisplacement);

	vec3 lightDir = normalize(lightPos - inFragPos);
	float diffuse = max(dot(combinedNormal, lightDir), 0.0);

	vec3 albedo = texture(albedoTexture, inUV).xyz;
	outFragColor = vec4((ambient + diffuse) * albedo, 1.0);
}
