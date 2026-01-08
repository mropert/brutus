#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive : require

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inFragPos;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBitangent;
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform texture2D textures[1024];
layout(set = 0, binding = 1) uniform sampler sampl;

#include "constants.glsl"
#include "mesh.glsl"

layout( push_constant, std430 ) uniform constants
{
    Mesh mesh;
    Material material;
    SceneConstants scene_constants;
} PushConstants;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    const float a      = roughness*roughness;
    const float a2     = a*a;
    const float NdotH  = max(dot(N, H), 0.0);
    const float NdotH2 = NdotH*NdotH;

    const float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    const float r = (roughness + 1.0);
    const float k = (r*r) / 8.0;

    const float num   = NdotV;
    const float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    const float NdotV = max(dot(N, V), 0.0);
    const float NdotL = max(dot(N, L), 0.0);
    const float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    const float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

void main_flat()
{
    const vec3 albedo = texture(sampler2D(textures[PushConstants.material.albedo_index], sampl), inUV).xyz * PushConstants.material.baseColor.xyz;
    outFragColor = vec4(albedo, 1.0);
}

void main_basic_diffuse()
{
    vec3 albedo = texture(sampler2D(textures[PushConstants.material.albedo_index], sampl), inUV).xyz * PushConstants.material.baseColor.xyz;
    const vec3 N = normalize(inNormal);
    const vec3 L = normalize(PushConstants.scene_constants.main_light.xyz);
    const float NdotL = max(dot(N, -L), 0.0);

    albedo = albedo * (NdotL + 0.1);
    outFragColor = vec4(albedo, 1.0);
}

void main()
{
    const vec3 albedo = texture(sampler2D(textures[PushConstants.material.albedo_index], sampl), inUV).xyz * PushConstants.material.baseColor.xyz;
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, PushConstants.material.metalness);

    const vec3 normalDisplacement = (texture(sampler2D(textures[PushConstants.material.normal_index], sampl), inUV).xyz * 2.0) - 1.0;
    const mat3 TBN = mat3(normalize(inTangent), normalize(inBitangent), normalize(inNormal));
    const vec3 N = normalize(TBN * normalDisplacement);

    const vec3 V = normalize(PushConstants.scene_constants.camera_position - inFragPos);

    const vec3 L = normalize(-PushConstants.scene_constants.main_light.xyz);
    const vec3 H = normalize(V + L);
    const float attenuation = PushConstants.scene_constants.main_light.w;
    const vec3 radiance     = vec3(1.0, 1.0, 1.0) * attenuation;

    // cook-torrance brdf
    const float NDF = DistributionGGX(N, H, PushConstants.material.roughness);
    const float G   = GeometrySmith(N, V, L, PushConstants.material.roughness);
    const vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

    const vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - PushConstants.material.metalness;

    const vec3 numerator    = NDF * G * F;
    const float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    const vec3 specular     = numerator / denominator;

    // add to outgoing radiance Lo
    const float NdotL = max(dot(N, L), 0.0);
    const vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    const vec3 ambient = vec3(0.03) * albedo;// * ao;
    const vec3 color = ambient + Lo;

    //color = color / (color + vec3(1.0));
    //color = pow(color, vec3(1.0/2.2));

    outFragColor = vec4(color, 1.0);
}
