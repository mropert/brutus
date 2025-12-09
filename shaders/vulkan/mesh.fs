#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inFragPos;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBitangent;
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform texture2D textures[1024];
layout(set = 0, binding = 1) uniform sampler sampl;

#include "constants.glsl"

layout(push_constant) uniform constants
{
	vec4 baseColor;
	float metalness;
	float roughness;
	uint albedo_index;
	uint normal_index;
	SceneConstants scene_constants;
} PushConstants;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a      = roughness*roughness;
	float a2     = a*a;
	float NdotH  = max(dot(N, H), 0.0);
	float NdotH2 = NdotH*NdotH;

	float num   = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;

	float num   = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2  = GeometrySchlickGGX(NdotV, roughness);
	float ggx1  = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

void main_flat()
{
	vec3 albedo = texture(sampler2D(textures[PushConstants.albedo_index], sampl), inUV).xyz * PushConstants.baseColor.xyz;
	outFragColor = vec4(albedo, 1.0);
}

void main_basic_diffuse()
{
	vec3 albedo = texture(sampler2D(textures[PushConstants.albedo_index], sampl), inUV).xyz * PushConstants.baseColor.xyz;
	vec3 N = normalize(inNormal);
	vec3 L = normalize(PushConstants.scene_constants.main_light.xyz);
	float NdotL = max(dot(N, -L), 0.0);

	albedo = albedo * (NdotL + 0.1);
	outFragColor = vec4(albedo, 1.0);
}

void main()
{
	vec3 albedo = texture(sampler2D(textures[PushConstants.albedo_index], sampl), inUV).xyz * PushConstants.baseColor.xyz;
	vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, PushConstants.metalness);

	vec3 normalDisplacement = (texture(sampler2D(textures[PushConstants.normal_index], sampl), inUV).xyz * 2.0) - 1.0;
	mat3 TBN = mat3(normalize(inTangent), normalize(inBitangent), normalize(inNormal));
	vec3 N = normalize(TBN * normalDisplacement);

	vec3 V = normalize(PushConstants.scene_constants.camera_position - inFragPos);

	vec3 L = normalize(-PushConstants.scene_constants.main_light.xyz);
	vec3 H = normalize(V + L);
	float attenuation = PushConstants.scene_constants.main_light.w;
	vec3 radiance     = vec3(1.0, 1.0, 1.0) * attenuation;

	// cook-torrance brdf
	float NDF = DistributionGGX(N, H, PushConstants.roughness);
	float G   = GeometrySmith(N, V, L, PushConstants.roughness);
	vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - PushConstants.metalness;

	vec3 numerator    = NDF * G * F;
	float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
	vec3 specular     = numerator / denominator;

	// add to outgoing radiance Lo
	float NdotL = max(dot(N, L), 0.0);
	vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    vec3 ambient = vec3(0.03) * albedo;// * ao;
    vec3 color = ambient + Lo;

    //color = color / (color + vec3(1.0));
    //color = pow(color, vec3(1.0/2.2));

    outFragColor = vec4(color, 1.0);
}
