#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_GOOGLE_include_directive : require

#include "constants.glsl"
#include "mesh.glsl"

layout( location = 0 ) in vec2 inUV;
layout( location = 1 ) in vec3 inFragPos;
layout( location = 2 ) in vec3 inNormal;
layout( location = 3 ) in vec4 inTangent;
layout( location = 4 ) in flat uint inMaterial;
layout( location = 0 ) out vec4 outFragColor;

layout( set = 0, binding = 0 ) uniform texture2D textures[ 1024 ];
layout( set = 0, binding = 2 ) uniform sampler samplers[ 8 ];
layout( std430, set = 1, binding = 1 ) readonly buffer ssbo2
{
	Material materials[];
};

layout( push_constant, std430 ) uniform block
{
	FrameConstants frame_constants;
};

#include "shadow.glsl"

const float PI = 3.14159265359;

vec3 fresnelSchlick( float cosTheta, vec3 F0 )
{
	return F0 + ( 1.0 - F0 ) * pow( clamp( 1.0 - cosTheta, 0.0, 1.0 ), 5.0 );
}

// Fresnel that folds in roughness, for the ambient/environment term
vec3 fresnelSchlickRoughness( float cosTheta, vec3 F0, float roughness )
{
	return F0 + ( max( vec3( 1.0 - roughness ), F0 ) - F0 ) * pow( clamp( 1.0 - cosTheta, 0.0, 1.0 ), 5.0 );
}

// Analytic environment BRDF approximation (Karis, "mobile") - stands in for the split-sum LUT
vec2 EnvBRDFApprox( float NdotV, float roughness )
{
	const vec4 c0 = vec4( -1.0, -0.0275, -0.572, 0.022 );
	const vec4 c1 = vec4( 1.0, 0.0425, 1.04, -0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( -9.28 * NdotV ) ) * r.x + r.y;
	return vec2( -1.04, 1.04 ) * a004 + r.zw;
}

float DistributionGGX( vec3 N, vec3 H, float roughness )
{
	const float a = roughness * roughness;
	const float a2 = a * a;
	const float NdotH = max( dot( N, H ), 0.0 );
	const float NdotH2 = NdotH * NdotH;

	const float num = a2;
	float denom = ( NdotH2 * ( a2 - 1.0 ) + 1.0 );
	denom = PI * denom * denom;

	return num / denom;
}

float GeometrySchlickGGX( float NdotV, float roughness )
{
	const float r = ( roughness + 1.0 );
	const float k = ( r * r ) / 8.0;

	const float num = NdotV;
	const float denom = NdotV * ( 1.0 - k ) + k;

	return num / denom;
}
float GeometrySmith( vec3 N, vec3 V, vec3 L, float roughness )
{
	const float NdotV = max( dot( N, V ), 0.0 );
	const float NdotL = max( dot( N, L ), 0.0 );
	const float ggx2 = GeometrySchlickGGX( NdotV, roughness );
	const float ggx1 = GeometrySchlickGGX( NdotL, roughness );

	return ggx1 * ggx2;
}

// Direct light attenuation for normal mapped shading at the shadow terminator, from Chiang, Li, Burley, "Taming the
// Shadow Terminator", SIGGRAPH 2019. 0 where the geometry faces away from the light, which the shading normal can't
// see, then rises smoothly to 1 as the geometry receives the light the shading normal assumes.
// geometric_normal is the interpolated vertex normal, facing the viewer for double sided surfaces.
float bump_terminator_shadowing( vec3 geometric_normal, vec3 shading_normal, vec3 to_light )
{
	const float NgL = dot( geometric_normal, to_light );
	if ( NgL <= 0.0 )
	{
		return 0.0;
	}
	const float NsL = dot( shading_normal, to_light );
	const float NgNs = dot( geometric_normal, shading_normal );
	const float G = min( 1.0, NgL / max( NsL * NgNs, 1e-4 ) );
	return -G * G * G + G * G + G;
}

// Ambient / fake-IBL: no cubemap yet, so approximate the environment with a hemisphere (sky above, ground below).
// These colors are the environment radiance, sampled with the normal for diffuse light and the reflection vector for
// specular. Tied to the sun so there's a single exposure knob (the sun) and the balance between them stays fixed.
// sky_to_sun is the "mood": higher = flatter/overcast.
vec3 hemisphere_environment( vec3 direction )
{
	const float intensity = frame_constants.main_light.w * frame_constants.sky_to_sun;
	const vec3 sky_color = vec3( 0.5, 0.6, 0.7 ) * intensity;
	const vec3 ground_color = vec3( 0.2, 0.18, 0.15 ) * intensity;
	return mix( ground_color, sky_color, 0.5 * ( direction.y + 1.0 ) );
}

void main_flat()
{
	const vec3 albedo = texture( sampler2D( textures[ materials[ inMaterial ].albedo_index ], samplers[ 0 ] ), inUV ).xyz
		* materials[ inMaterial ].baseColor.xyz;
	outFragColor = vec4( albedo, 1.0 );
}

void main_basic_diffuse()
{
	vec3 albedo = texture( sampler2D( textures[ materials[ inMaterial ].albedo_index ], samplers[ 0 ] ), inUV ).xyz
		* materials[ inMaterial ].baseColor.xyz;
	const vec3 N = normalize( inNormal );
	const vec3 L = normalize( frame_constants.main_light.xyz );
	const float NdotL = max( dot( N, -L ), 0.0 );

	albedo = albedo * ( NdotL + 0.1 );
	outFragColor = vec4( albedo, 1.0 );
}

void main()
{
	const uint albedo_index = materials[ inMaterial ].albedo_index;
	// If albedo is missing just display normal as a debug fallback
	if ( albedo_index == -1 )
	{
		outFragColor = vec4( inNormal, 1.0 );
		return;
	}

	const vec4 albedo = texture( sampler2D( textures[ materials[ inMaterial ].albedo_index ], samplers[ 0 ] ), inUV )
		* materials[ inMaterial ].baseColor;
	if ( USE_ALPHA && albedo.a < materials[ inMaterial ].alpha_cutoff )
	{
		discard;
	}

	const vec3 L = normalize( -frame_constants.main_light.xyz );
	vec3 N = normalize( inNormal );
	const float facing = gl_FrontFacing ? 1.0 : -1.0;
	const vec3 geometric_normal = N * facing;

	const float shadow = compute_shadow( frame_constants, inFragPos, geometric_normal, L );

	if ( materials[ inMaterial ].normal_index != -1 )
	{
		const vec3 normalDisplacement = ( texture( sampler2D( textures[ materials[ inMaterial ].normal_index ], samplers[ 0 ] ), inUV ).xyz
										  * 2.0 )
			- 1.0;
		const vec3 tangent = normalize( inTangent.xyz );
		const mat3 TBN = mat3( tangent, cross( N, tangent * inTangent.w ), normalize( inNormal ) );
		N = normalize( TBN * normalDisplacement );
	}
	N *= facing;

	const float NdotL = max( dot( N, L ), 0.0 );
	const float direct_light = shadow * bump_terminator_shadowing( geometric_normal, N, L );
	const float ambient_factor = 0.1;
	vec3 color;

	if ( frame_constants.pbr == 1 )
	{
		const vec3 V = normalize( frame_constants.camera_position - inFragPos );
		const vec3 H = normalize( V + L );
		const float NdotV = max( dot( N, V ), 1e-4 );
		const float HdotV = max( dot( H, V ), 1e-4 );
		const float attenuation = frame_constants.main_light.w;
		const vec3 radiance = vec3( 1.0, 1.0, 1.0 ) * attenuation;

		vec3 diffuse = albedo.rgb;
		vec3 F0;
		float roughness;
		if ( materials[ inMaterial ].workflow == MATERIAL_WORKFLOW_METALLIC_ROUGHNESS )
		{
			// cook-torrance brdf
			roughness = materials[ inMaterial ].roughness;
			float metalness = materials[ inMaterial ].metalness;
			if ( materials[ inMaterial ].metallic_roughness_index != -1 )
			{
				const vec3 metallic_roughness = texture( sampler2D( textures[ materials[ inMaterial ].metallic_roughness_index ],
																	samplers[ 0 ] ),
														 inUV )
													.rgb;
				metalness *= metallic_roughness.b;
				roughness *= metallic_roughness.g;
			}
			float specular_factor = 1.0;
			if ( materials[ inMaterial ].specular_index != -1 )
			{
				specular_factor = texture( sampler2D( textures[ materials[ inMaterial ].specular_index ], samplers[ 0 ] ), inUV ).a;
			}

			diffuse *= ( 1 - metalness );
			F0 = mix( vec3( 0.04 ), albedo.rgb, metalness ) * specular_factor;
		}
		else // MATERIAL_WORKFLOW_SPECULAR_GLOSSINESS
		{
			vec4 specular_glossiness = materials[ inMaterial ].specular_glossiness;
			if ( materials[ inMaterial ].specular_glossiness_index != -1 )
			{
				specular_glossiness *= texture( sampler2D( textures[ materials[ inMaterial ].specular_glossiness_index ], samplers[ 0 ] ),
												inUV );
			}
			diffuse *= ( 1 - max( max( specular_glossiness.r, specular_glossiness.g ), specular_glossiness.b ) );
			F0 = specular_glossiness.rgb;
			roughness = 1 - specular_glossiness.a;
		}

		// Even a perfect mirror doesn't have 0 roughness
		// This ensures GGX doesn't degenerate to infinity at some angles
		roughness = max( roughness, 0.045 );

		const vec3 F = fresnelSchlick( HdotV, F0 );
		const float G = GeometrySmith( N, V, L, roughness );
		const float NDF = DistributionGGX( N, H, roughness );
		const vec3 numerator = NDF * G * F;
		const float denominator = max( 4.0 * NdotV * NdotL + 1e-4, 1e-4 );
		const vec3 specular = numerator / denominator;

		// add to outgoing radiance Lo
		const vec3 Lo = ( ( 1 - F ) * diffuse / PI + specular ) * radiance * NdotL * direct_light;

		const vec3 R = reflect( -V, N );
		const vec3 irradiance = hemisphere_environment( N ); // diffuse env
		const vec3 prefiltered = hemisphere_environment( R ); // specular env

		const vec3 F_amb = fresnelSchlickRoughness( NdotV, F0, roughness );
		const vec2 envBRDF = EnvBRDFApprox( NdotV, roughness );
		const vec3 diffuse_ambient = irradiance * diffuse;
		const vec3 specular_ambient = prefiltered * ( F0 * envBRDF.x + envBRDF.y );
		const vec3 ambient = ( 1.0 - F_amb ) * diffuse_ambient + specular_ambient; // * ao;

		color = ambient + Lo;
	}
	else
	{
		color = albedo.rgb * ( NdotL * direct_light * frame_constants.main_light.w * 0.5 + hemisphere_environment( N ) );
	}

	outFragColor = vec4( color * get_shadow_cascade_tint( frame_constants, inFragPos ), albedo.a );
}
