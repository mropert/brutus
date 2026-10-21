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
	SceneConstants scene_constants;
};

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
	const vec3 L = normalize( scene_constants.main_light.xyz );
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
	if ( USE_ALPHA && albedo.a < 0.5 )
	{
		discard;
	}

	const vec3 L = normalize( -scene_constants.main_light.xyz );
	vec3 N = normalize( inNormal );
	const float facing = gl_FrontFacing ? 1.0 : -1.0;

	// Shadow
	float shadow = 1.0;
	if ( scene_constants.shadow_map != -1 )
	{
		const float NdotL_geo = dot( N * facing, L );
		if ( NdotL_geo > 0.0 )
		{
			const vec4 shadow_pos = scene_constants.shadow_view_projection_matrix * vec4( inFragPos, 1.0 );
			const vec3 shadow_ndc = shadow_pos.xyz / shadow_pos.w;
			const vec2 shadow_uv = vec2( shadow_ndc.x * 0.5 + 0.5, 0.5 - shadow_ndc.y * 0.5 );
			if ( all( greaterThanEqual( shadow_uv, vec2( 0 ) ) ) && all( lessThanEqual( shadow_uv, vec2( 1 ) ) ) && shadow_ndc.z >= 0
				 && shadow_ndc.z <= 1.0 )
			{
				const float shadow_bias = max( 0.0005 * ( 1.0 - dot( N * facing, L ) ), 0.0005 );
				shadow = texture( sampler2DShadow( textures[ scene_constants.shadow_map ], samplers[ 2 ] ),
								  vec3( shadow_uv, shadow_ndc.z - shadow_bias ) );
				const vec2 edge = min( shadow_uv, 1.0 - shadow_uv );
				const float fade = smoothstep( 0.0, 0.1, min( edge.x, edge.y ) );
				shadow = mix( 1.0, shadow, fade );
			}
		}
	}

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
	const float ambient_factor = 0.1;
	vec3 color;

	if ( scene_constants.pbr == 1 )
	{
		const vec3 V = normalize( scene_constants.camera_position - inFragPos );
		const vec3 H = normalize( V + L );
		const float NdotV = max( dot( N, V ), 1e-4 );
		const float HdotV = max( dot( H, V ), 1e-4 );
		const float attenuation = scene_constants.main_light.w;
		const vec3 radiance = vec3( 1.0, 1.0, 1.0 ) * attenuation;

		// cook-torrance brdf
		float roughness = materials[ inMaterial ].roughness;
		float metalness = materials[ inMaterial ].metalness;
		if ( materials[ inMaterial ].metallic_roughness_index != -1 )
		{
			const vec3 metallic_roughness = texture(
												sampler2D( textures[ materials[ inMaterial ].metallic_roughness_index ], samplers[ 0 ] ),
												inUV )
												.rgb;
			// Our current test model has broken data on the MR texture
			metalness = 0;
			roughness *= metallic_roughness.b;
		}
		float specular_factor = 1.0;
		if ( materials[ inMaterial ].specular_index != -1 )
		{
			specular_factor = texture( sampler2D( textures[ materials[ inMaterial ].specular_index ], samplers[ 0 ] ), inUV ).a;
		}

		const float NDF = DistributionGGX( N, H, roughness );
		const float G = GeometrySmith( N, V, L, roughness );
		const vec3 F0 = mix( vec3( 0.04 ), albedo.rgb, metalness ) * specular_factor;
		const vec3 F = fresnelSchlick( HdotV, F0 );

		const vec3 numerator = NDF * G * F;
		const float denominator = max( 4.0 * NdotV * NdotL + 1e-4, 1e-4 );
		const vec3 specular = numerator / denominator;

		// add to outgoing radiance Lo
		const vec3 kD = ( 1 - F ) * ( 1 - metalness );
		const vec3 Lo = ( kD * albedo.rgb / PI + specular ) * radiance * NdotL * shadow;

		// Ambient / fake-IBL: no cubemap yet, so approximate the environment with a
		// hemisphere (sky above, ground below). These colors ARE the environment radiance.
		// Tie the sky/ambient to the sun so there's a single exposure knob (the sun) and the
		// balance between them stays fixed. sky_to_sun is the "mood": higher = flatter/overcast.
		const float env_intensity = attenuation * scene_constants.sky_to_sun;
		const vec3 sky_color = vec3( 0.5, 0.6, 0.7 ) * env_intensity;
		const vec3 ground_color = vec3( 0.2, 0.18, 0.15 ) * env_intensity;
		const vec3 R = reflect( -V, N );
		const vec3 irradiance = mix( ground_color, sky_color, 0.5 * ( N.y + 1.0 ) ); // diffuse env
		const vec3 prefiltered = mix( ground_color, sky_color, 0.5 * ( R.y + 1.0 ) ); // specular env

		const vec3 F_amb = fresnelSchlickRoughness( NdotV, F0, roughness );
		const vec3 kD_amb = ( 1.0 - F_amb ) * ( 1.0 - metalness );
		const vec2 envBRDF = EnvBRDFApprox( NdotV, roughness );
		const vec3 diffuse_ambient = irradiance * albedo.rgb;
		const vec3 specular_ambient = prefiltered * ( F0 * envBRDF.x + envBRDF.y );
		const vec3 ambient = kD_amb * diffuse_ambient + specular_ambient; // * ao;

		color = ambient + Lo;
	}
	else
	{
		color = albedo.rgb * ( NdotL * shadow + scene_constants.sky_to_sun ) * scene_constants.main_light.w * 0.5;
	}

	outFragColor = vec4( color, albedo.a );
}
