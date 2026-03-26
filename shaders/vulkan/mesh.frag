#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive : require

#include "constants.glsl"
#include "mesh.glsl"

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inFragPos;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBitangent;
layout (location = 5) in flat uint inMaterial;
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform texture2D textures[1024];
layout(set = 0, binding = 2) uniform sampler samplers[8];
layout(std430, set = 1, binding = 1) readonly buffer ssbo2 { Material materials[]; };

layout( push_constant, std430 ) uniform block
{
    SceneConstants scene_constants;
};

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
    const vec3 albedo = texture(sampler2D(textures[materials[inMaterial].albedo_index], samplers[0]), inUV).xyz * materials[inMaterial].baseColor.xyz;
    outFragColor = vec4(albedo, 1.0);
}

void main_basic_diffuse()
{
    vec3 albedo = texture(sampler2D(textures[materials[inMaterial].albedo_index], samplers[0]), inUV).xyz * materials[inMaterial].baseColor.xyz;
    const vec3 N = normalize(inNormal);
    const vec3 L = normalize(scene_constants.main_light.xyz);
    const float NdotL = max(dot(N, -L), 0.0);

    albedo = albedo * (NdotL + 0.1);
    outFragColor = vec4(albedo, 1.0);
}

void main()
{
    const uint albedo_index = materials[inMaterial].albedo_index;
    // If albedo is missing just display normal as a debug fallback
    if (albedo_index == 0)
    {
        outFragColor = vec4(inNormal, 1.0);
        return;
    }

    const vec3 albedo = texture(sampler2D(textures[materials[inMaterial].albedo_index], samplers[0]), inUV).xyz * materials[inMaterial].baseColor.xyz;
    vec3 N;
    if ( materials[inMaterial].normal_index != -1 )
    {
        const vec3 normalDisplacement = (texture(sampler2D(textures[materials[inMaterial].normal_index], samplers[0]), inUV).xyz * 2.0) - 1.0;
        const mat3 TBN = mat3(normalize(inTangent), normalize(inBitangent), normalize(inNormal));
        N = normalize(TBN * normalDisplacement);
    }
    else
    {
        N = normalize(inNormal);
    }

    const vec3 L = normalize(-scene_constants.main_light.xyz);
    const float NdotL = max(dot(N, L), 0.0);
    const float ambient_factor = 0.03;
    vec3 color;

    if ( scene_constants.pbr == 1 )
    {
        const vec3 V = normalize(scene_constants.camera_position - inFragPos);
        const vec3 H = normalize(V + L);
        const float  NdotV = max(dot(N, V), 1e-4);
        const float  HdotV = max(dot(H, V), 1e-4);
        const float attenuation = scene_constants.main_light.w;
        const vec3 radiance     = vec3(1.0, 1.0, 1.0) * attenuation;

        // cook-torrance brdf
        float roughness = materials[inMaterial].roughness;
        float metalness = materials[inMaterial].metalness;
        if ( materials[inMaterial].metallic_roughness_index != -1 )
        {
            const vec3 metallic_roughness = texture(sampler2D(textures[materials[inMaterial].metallic_roughness_index], samplers[0]), inUV).rgb;
            metalness *= metallic_roughness.b;
            roughness *= metallic_roughness.g;
        }
        const float NDF = DistributionGGX(N, H, roughness);
        const float G   = GeometrySmith(N, V, L, roughness);
        const vec3 F0 = mix(vec3(0.04), albedo, metalness);
        const vec3 F = fresnelSchlick(HdotV, F0);

        const vec3 numerator    = NDF * G * F;
        const float denominator = max(4.0 * NdotV * HdotV + 1e-4, 1e-4);
        const vec3 specular     = numerator / denominator;

        // add to outgoing radiance Lo
        const vec3 kD = (1 - F) * (1 - metalness);
        const vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

        const vec3 ambient = ambient_factor * albedo;// * ao;
        color = ambient + Lo;

        //color = color / (color + vec3(1.0));
        //color = pow(color, vec3(1.0/2.2));
    }
    else
    {
        color = albedo * (NdotL + ambient_factor);
    }

    outFragColor = vec4(color, 1.0);
}
