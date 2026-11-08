// BC1 block encoder. Include only: no bindings, no main.
//
// From Betsy (https://github.com/darksylinc/betsy), commit cc723dc, bin/Data/bc1.glsl and src/betsy/BC1_tables.inl.
//
// Copyright 2020-2022 Matias N. Goldberg
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Based on stb_dxt (https://github.com/nothings/stb/blob/master/stb_dxt.h), original by fabian "ryg" giesen, ported to C
// by Sean Barrett, MIT (Copyright (c) 2017 Sean Barrett) or public domain. Adapted to a compute shader by Matias N.
// Goldberg.
//
// Changes from Betsy: a function per block instead of a shader with its own bindings and main, GLSL types instead of
// HLSL style macros, match tables as constants instead of an SSBO, no dithering, the single color path rounds instead
// of truncating texels back to 8 bits.

const float BC1_FLT_MAX = 340282346638528859811704183484516925440.0f;

// Best endpoints to reproduce one 8 bit value: max in the low byte, min in the next. From stb_dxt's stb__OMatch5 and
// stb__OMatch6.
// clang-format off
const uint BC1_OMATCH5[ 256 ] = uint[ 256 ](
	0x0000u, 0x0000u, 0x0100u, 0x0100u, 0x0001u, 0x0001u, 0x0001u, 0x0101u,
	0x0101u, 0x0002u, 0x0002u, 0x0400u, 0x0102u, 0x0102u, 0x0102u, 0x0003u,
	0x0003u, 0x0003u, 0x0103u, 0x0501u, 0x0203u, 0x0203u, 0x0004u, 0x0004u,
	0x0104u, 0x0104u, 0x0204u, 0x0204u, 0x0204u, 0x0503u, 0x0105u, 0x0105u,
	0x0205u, 0x0404u, 0x0305u, 0x0305u, 0x0305u, 0x0206u, 0x0206u, 0x0206u,
	0x0306u, 0x0505u, 0x0406u, 0x0406u, 0x0804u, 0x0307u, 0x0307u, 0x0307u,
	0x0407u, 0x0407u, 0x0407u, 0x0507u, 0x0905u, 0x0607u, 0x0607u, 0x0408u,
	0x0408u, 0x0508u, 0x0508u, 0x0608u, 0x0608u, 0x0608u, 0x0907u, 0x0509u,
	0x0509u, 0x0609u, 0x0808u, 0x0709u, 0x0709u, 0x0709u, 0x060au, 0x060au,
	0x060au, 0x070au, 0x0909u, 0x080au, 0x080au, 0x0c08u, 0x070bu, 0x070bu,
	0x070bu, 0x080bu, 0x080bu, 0x080bu, 0x090bu, 0x0d09u, 0x0a0bu, 0x0a0bu,
	0x080cu, 0x080cu, 0x090cu, 0x090cu, 0x0a0cu, 0x0a0cu, 0x0a0cu, 0x0d0bu,
	0x090du, 0x090du, 0x0a0du, 0x0c0cu, 0x0b0du, 0x0b0du, 0x0b0du, 0x0a0eu,
	0x0a0eu, 0x0a0eu, 0x0b0eu, 0x0d0du, 0x0c0eu, 0x0c0eu, 0x100cu, 0x0b0fu,
	0x0b0fu, 0x0b0fu, 0x0c0fu, 0x0c0fu, 0x0c0fu, 0x0d0fu, 0x110du, 0x0e0fu,
	0x0e0fu, 0x0c10u, 0x0c10u, 0x0d10u, 0x0d10u, 0x0e10u, 0x0e10u, 0x0e10u,
	0x110fu, 0x0d11u, 0x0d11u, 0x0e11u, 0x1010u, 0x0f11u, 0x0f11u, 0x0f11u,
	0x0e12u, 0x0e12u, 0x0e12u, 0x0f12u, 0x1111u, 0x1012u, 0x1012u, 0x1410u,
	0x0f13u, 0x0f13u, 0x0f13u, 0x1013u, 0x1013u, 0x1013u, 0x1113u, 0x1511u,
	0x1213u, 0x1213u, 0x1014u, 0x1014u, 0x1114u, 0x1114u, 0x1214u, 0x1214u,
	0x1214u, 0x1513u, 0x1115u, 0x1115u, 0x1215u, 0x1414u, 0x1315u, 0x1315u,
	0x1315u, 0x1216u, 0x1216u, 0x1216u, 0x1316u, 0x1515u, 0x1416u, 0x1416u,
	0x1814u, 0x1317u, 0x1317u, 0x1317u, 0x1417u, 0x1417u, 0x1417u, 0x1517u,
	0x1915u, 0x1617u, 0x1617u, 0x1418u, 0x1418u, 0x1518u, 0x1518u, 0x1618u,
	0x1618u, 0x1618u, 0x1917u, 0x1519u, 0x1519u, 0x1619u, 0x1818u, 0x1719u,
	0x1719u, 0x1719u, 0x161au, 0x161au, 0x161au, 0x171au, 0x1919u, 0x181au,
	0x181au, 0x1c18u, 0x171bu, 0x171bu, 0x171bu, 0x181bu, 0x181bu, 0x181bu,
	0x191bu, 0x1d19u, 0x1a1bu, 0x1a1bu, 0x181cu, 0x181cu, 0x191cu, 0x191cu,
	0x1a1cu, 0x1a1cu, 0x1a1cu, 0x1d1bu, 0x191du, 0x191du, 0x1a1du, 0x1c1cu,
	0x1b1du, 0x1b1du, 0x1b1du, 0x1a1eu, 0x1a1eu, 0x1a1eu, 0x1b1eu, 0x1d1du,
	0x1c1eu, 0x1c1eu, 0x1c1eu, 0x1b1fu, 0x1b1fu, 0x1b1fu, 0x1c1fu, 0x1c1fu,
	0x1c1fu, 0x1d1fu, 0x1d1fu, 0x1e1fu, 0x1e1fu, 0x1e1fu, 0x1f1fu, 0x1f1fu
);

const uint BC1_OMATCH6[ 256 ] = uint[ 256 ](
	0x0000u, 0x0100u, 0x0001u, 0x0001u, 0x0101u, 0x0002u, 0x0102u, 0x0003u,
	0x0003u, 0x0103u, 0x0004u, 0x0004u, 0x0104u, 0x0005u, 0x0105u, 0x0006u,
	0x0006u, 0x0106u, 0x0007u, 0x0007u, 0x0107u, 0x0008u, 0x0108u, 0x0108u,
	0x0208u, 0x0109u, 0x0209u, 0x0209u, 0x0309u, 0x020au, 0x030au, 0x030au,
	0x040au, 0x030bu, 0x040bu, 0x040bu, 0x050bu, 0x040cu, 0x050cu, 0x050cu,
	0x060cu, 0x050du, 0x060du, 0x1008u, 0x070du, 0x060eu, 0x070eu, 0x1109u,
	0x080eu, 0x070fu, 0x080fu, 0x100bu, 0x090fu, 0x0a0fu, 0x0810u, 0x0910u,
	0x0a10u, 0x0d0fu, 0x0911u, 0x0a11u, 0x0b11u, 0x100fu, 0x0a12u, 0x0b12u,
	0x0c12u, 0x1010u, 0x0b13u, 0x0c13u, 0x0d13u, 0x1111u, 0x0c14u, 0x0d14u,
	0x0e14u, 0x1013u, 0x0d15u, 0x0e15u, 0x0f15u, 0x1114u, 0x0e16u, 0x0f16u,
	0x0a19u, 0x1016u, 0x0f17u, 0x1017u, 0x0b1au, 0x1117u, 0x1018u, 0x1118u,
	0x0c1bu, 0x1218u, 0x1119u, 0x1219u, 0x0d1cu, 0x1319u, 0x121au, 0x131au,
	0x0e1du, 0x141au, 0x131bu, 0x141bu, 0x0f1eu, 0x151bu, 0x141cu, 0x151cu,
	0x151cu, 0x161cu, 0x151du, 0x161du, 0x2018u, 0x171du, 0x161eu, 0x171eu,
	0x2119u, 0x181eu, 0x171fu, 0x181fu, 0x201bu, 0x191fu, 0x1a1fu, 0x1820u,
	0x1920u, 0x1a20u, 0x1d1fu, 0x1921u, 0x1a21u, 0x1b21u, 0x201fu, 0x1a22u,
	0x1b22u, 0x1c22u, 0x2020u, 0x1b23u, 0x1c23u, 0x1d23u, 0x2121u, 0x1c24u,
	0x1d24u, 0x1e24u, 0x2023u, 0x1d25u, 0x1e25u, 0x1f25u, 0x2124u, 0x1e26u,
	0x1f26u, 0x1a29u, 0x2026u, 0x1f27u, 0x2027u, 0x1b2au, 0x2127u, 0x2028u,
	0x2128u, 0x1c2bu, 0x2228u, 0x2129u, 0x2229u, 0x1d2cu, 0x2329u, 0x222au,
	0x232au, 0x1e2du, 0x242au, 0x232bu, 0x242bu, 0x1f2eu, 0x252bu, 0x242cu,
	0x252cu, 0x252cu, 0x262cu, 0x252du, 0x262du, 0x3028u, 0x272du, 0x262eu,
	0x272eu, 0x3129u, 0x282eu, 0x272fu, 0x282fu, 0x302bu, 0x292fu, 0x2a2fu,
	0x2830u, 0x2930u, 0x2a30u, 0x2d2fu, 0x2931u, 0x2a31u, 0x2b31u, 0x302fu,
	0x2a32u, 0x2b32u, 0x2c32u, 0x3030u, 0x2b33u, 0x2c33u, 0x2d33u, 0x3131u,
	0x2c34u, 0x2d34u, 0x2e34u, 0x3033u, 0x2d35u, 0x2e35u, 0x2f35u, 0x3134u,
	0x2e36u, 0x2f36u, 0x2a39u, 0x3036u, 0x2f37u, 0x3037u, 0x2b3au, 0x3137u,
	0x3038u, 0x3138u, 0x2c3bu, 0x3238u, 0x3139u, 0x3239u, 0x2d3cu, 0x3339u,
	0x323au, 0x333au, 0x2e3du, 0x343au, 0x333bu, 0x343bu, 0x2f3eu, 0x353bu,
	0x343cu, 0x353cu, 0x353cu, 0x363cu, 0x353du, 0x363du, 0x363du, 0x373du,
	0x363eu, 0x373eu, 0x373eu, 0x383eu, 0x373fu, 0x383fu, 0x383fu, 0x393fu,
	0x3a3fu, 0x3b3fu, 0x3b3fu, 0x3c3fu, 0x3d3fu, 0x3e3fu, 0x3e3fu, 0x3f3fu
);
// clang-format on

// Returns the max (x) and min (y) RGB565 endpoints for a single color
vec2 bc1_single_color_endpoints( uvec3 rgb )
{
	const uvec3 entries = uvec3( BC1_OMATCH5[ rgb.r ], BC1_OMATCH6[ rgb.g ], BC1_OMATCH5[ rgb.b ] );
	const vec3 to_565 = vec3( 2048.0f, 32.0f, 1.0f );
	return vec2( dot( vec3( entries & 0xFFu ), to_565 ), dot( vec3( entries >> 8u ), to_565 ) );
}

vec3 rgb565to888( float rgb565 )
{
	vec3 retVal;
	retVal.x = floor( rgb565 / 2048.0f );
	retVal.y = floor( mod( rgb565, 2048.0f ) / 32.0f );
	retVal.z = floor( mod( rgb565, 32.0f ) );

	// This is the correct 565 to 888 conversion:
	//		rgb = floor( rgb * ( 255.0f / vec3( 31.0f, 63.0f, 31.0f ) ) + 0.5f )
	//
	// However stb_dxt follows a different one:
	//		rb = floor( rb * ( 256 / 32 + 8 / 32 ) );
	//		g  = floor( g  * ( 256 / 64 + 4 / 64 ) );
	//
	// I'm not sure exactly why but it's possible this is how the S3TC specifies it should be decoded
	// It's quite possible this is the reason:
	//		http://www.ludicon.com/castano/blog/2009/03/gpu-dxt-decompression/
	//
	// Or maybe it's just because it's cheap to do with integer shifts.
	// Anyway, we follow stb_dxt's conversion just in case
	// (gives almost the same result, with 1 or -1 of difference for a very few values)
	//
	// Perhaps when we make 888 -> 565 -> 888 it doesn't matter
	// because they end up mapping to the original number

	return floor( retVal * vec3( 8.25f, 4.0625f, 8.25f ) );
}

float rgb888to565( vec3 rgbValue )
{
	rgbValue.rb = floor( rgbValue.rb * 31.0f / 255.0f + 0.5f );
	rgbValue.g = floor( rgbValue.g * 63.0f / 255.0f + 0.5f );

	return rgbValue.r * 2048.0f + rgbValue.g * 32.0f + rgbValue.b;
}

// linear interpolation at 1/3 point between a and b, using desired rounding type
vec3 lerp13( vec3 a, vec3 b )
{
#ifdef STB_DXT_USE_ROUNDING_BIAS
	// with rounding bias
	return a + floor( ( b - a ) * ( 1.0f / 3.0f ) + 0.5f );
#else
	// without rounding bias
	return floor( ( 2.0f * a + b ) / 3.0f );
#endif
}

/// Unpacks a block of 4 colours from two 16-bit endpoints
void EvalColors( out vec3 colours[ 4 ], float c0, float c1 )
{
	colours[ 0 ] = rgb565to888( c0 );
	colours[ 1 ] = rgb565to888( c1 );
	colours[ 2 ] = lerp13( colours[ 0 ], colours[ 1 ] );
	colours[ 3 ] = lerp13( colours[ 1 ], colours[ 0 ] );
}

/** The color optimization function. (Clever code, part 1)
@param outMinEndp16 [out]
	Minimum endpoint, in RGB565
@param outMaxEndp16 [out]
	Maximum endpoint, in RGB565
*/
void OptimizeColorsBlock( const uint srcPixelsBlock[ 16 ], out float outMinEndp16, out float outMaxEndp16 )
{
	// determine color distribution
	vec3 avgColour;
	vec3 minColour;
	vec3 maxColour;

	avgColour = minColour = maxColour = unpackUnorm4x8( srcPixelsBlock[ 0 ] ).xyz;
	for ( int i = 1; i < 16; ++i )
	{
		const vec3 currColourUnorm = unpackUnorm4x8( srcPixelsBlock[ i ] ).xyz;
		avgColour += currColourUnorm;
		minColour = min( minColour, currColourUnorm );
		maxColour = max( maxColour, currColourUnorm );
	}

	avgColour = round( avgColour * 255.0f / 16.0f );
	maxColour *= 255.0f;
	minColour *= 255.0f;

	// determine covariance matrix
	float cov[ 6 ];
	for ( int i = 0; i < 6; ++i )
		cov[ i ] = 0;

	for ( int i = 0; i < 16; ++i )
	{
		const vec3 currColour = unpackUnorm4x8( srcPixelsBlock[ i ] ).xyz * 255.0f;
		vec3 rgbDiff = currColour - avgColour;

		cov[ 0 ] += rgbDiff.r * rgbDiff.r;
		cov[ 1 ] += rgbDiff.r * rgbDiff.g;
		cov[ 2 ] += rgbDiff.r * rgbDiff.b;
		cov[ 3 ] += rgbDiff.g * rgbDiff.g;
		cov[ 4 ] += rgbDiff.g * rgbDiff.b;
		cov[ 5 ] += rgbDiff.b * rgbDiff.b;
	}

	// convert covariance matrix to float, find principal axis via power iter
	for ( int i = 0; i < 6; ++i )
		cov[ i ] /= 255.0f;

	vec3 vF = maxColour - minColour;

	const int nIterPower = 4;
	for ( int iter = 0; iter < nIterPower; ++iter )
	{
		const float r = vF.r * cov[ 0 ] + vF.g * cov[ 1 ] + vF.b * cov[ 2 ];
		const float g = vF.r * cov[ 1 ] + vF.g * cov[ 3 ] + vF.b * cov[ 4 ];
		const float b = vF.r * cov[ 2 ] + vF.g * cov[ 4 ] + vF.b * cov[ 5 ];

		vF.r = r;
		vF.g = g;
		vF.b = b;
	}

	float magn = max( abs( vF.r ), max( abs( vF.g ), abs( vF.b ) ) );
	vec3 v;

	if ( magn < 4.0f )
	{ // too small, default to luminance
		v.r = 299.0f; // JPEG YCbCr luma coefs, scaled by 1000.
		v.g = 587.0f;
		v.b = 114.0f;
	}
	else
	{
		v = trunc( vF * ( 512.0f / magn ) );
	}

	// Pick colors at extreme points
	vec3 minEndpoint, maxEndpoint;
	float minDot = BC1_FLT_MAX;
	float maxDot = -BC1_FLT_MAX;
	for ( int i = 0; i < 16; ++i )
	{
		const vec3 currColour = unpackUnorm4x8( srcPixelsBlock[ i ] ).xyz * 255.0f;
		const float dotValue = dot( currColour, v );

		if ( dotValue < minDot )
		{
			minDot = dotValue;
			minEndpoint = currColour;
		}

		if ( dotValue > maxDot )
		{
			maxDot = dotValue;
			maxEndpoint = currColour;
		}
	}

	outMinEndp16 = rgb888to565( minEndpoint );
	outMaxEndp16 = rgb888to565( maxEndpoint );
}

// The color matching function
uint MatchColorsBlock( const uint srcPixelsBlock[ 16 ], vec3 colour[ 4 ] )
{
	uint mask = 0u;
	vec3 dir = colour[ 0 ] - colour[ 1 ];
	float stops[ 4 ];

	for ( int i = 0; i < 4; ++i )
		stops[ i ] = dot( colour[ i ], dir );

	// think of the colors as arranged on a line; project point onto that line, then choose
	// next color out of available ones. we compute the crossover points for "best color in top
	// half"/"best in bottom half" and then the same inside that subinterval.
	//
	// relying on this 1d approximation isn't always optimal in terms of euclidean distance,
	// but it's very close and a lot faster.
	// http://cbloomrants.blogspot.com/2008/12/12-08-08-dxtc-summary.html

	float c0Point = trunc( ( stops[ 1 ] + stops[ 3 ] ) * 0.5f );
	float halfPoint = trunc( ( stops[ 3 ] + stops[ 2 ] ) * 0.5f );
	float c3Point = trunc( ( stops[ 2 ] + stops[ 0 ] ) * 0.5f );

	for ( uint i = 16u; i-- > 0u; )
	{
		const vec3 currColour = unpackUnorm4x8( srcPixelsBlock[ i ] ).xyz * 255.0f;

		const float dotValue = dot( currColour, dir );
		mask <<= 2u;

		if ( dotValue < halfPoint )
			mask |= ( ( dotValue < c0Point ) ? 1u : 3u );
		else
			mask |= ( ( dotValue < c3Point ) ? 2u : 0u );
	}

	return mask;
}

// The refinement function. (Clever code, part 2)
// Tries to optimize colors to suit block contents better.
// (By solving a least squares system via normal equations+Cramer's rule)
bool RefineBlock( const uint srcPixelsBlock[ 16 ], uint mask, inout float inOutMinEndp16, inout float inOutMaxEndp16 )
{
	float newMin16, newMax16;
	const float oldMin = inOutMinEndp16;
	const float oldMax = inOutMaxEndp16;

	if ( ( mask ^ ( mask << 2u ) ) < 4u ) // all pixels have the same index?
	{
		// yes, linear system would be singular; solve using optimal
		// single-color match on average color
		vec3 rgbVal = vec3( 8.0f / 255.0f, 8.0f / 255.0f, 8.0f / 255.0f );
		for ( int i = 0; i < 16; ++i )
			rgbVal += unpackUnorm4x8( srcPixelsBlock[ i ] ).xyz;

		rgbVal = floor( rgbVal * ( 255.0f / 16.0f ) );

		const vec2 endpoints = bc1_single_color_endpoints( uvec3( rgbVal ) );
		newMax16 = endpoints.x;
		newMin16 = endpoints.y;
	}
	else
	{
		const float w1Tab[ 4 ] = { 3, 0, 2, 1 };
		const float prods[ 4 ] = { 589824.0f, 2304.0f, 262402.0f, 66562.0f };
		// ^some magic to save a lot of multiplies in the accumulating loop...
		// (precomputed products of weights for least squares system, accumulated inside one 32-bit
		// register)

		float akku = 0.0f;
		uint cm = mask;
		vec3 at1 = vec3( 0, 0, 0 );
		vec3 at2 = vec3( 0, 0, 0 );
		for ( int i = 0; i < 16; ++i, cm >>= 2u )
		{
			const vec3 currColour = unpackUnorm4x8( srcPixelsBlock[ i ] ).xyz * 255.0f;

			const uint step = cm & 3u;
			const float w1 = w1Tab[ step ];
			akku += prods[ step ];
			at1 += currColour * w1;
			at2 += currColour;
		}

		at2 = 3.0f * at2 - at1;

		// extract solutions and decide solvability
		const float xx = floor( akku / 65535.0f );
		const float yy = floor( mod( akku, 65535.0f ) / 256.0f );
		const float xy = mod( akku, 256.0f );

		vec2 f_rb_g;
		f_rb_g.x = 3.0f * 31.0f / 255.0f / ( xx * yy - xy * xy );
		f_rb_g.y = f_rb_g.x * 63.0f / 31.0f;

		// solve.
		const vec3 newMaxVal = clamp( floor( ( at1 * yy - at2 * xy ) * f_rb_g.xyx + 0.5f ), vec3( 0.0f, 0.0f, 0.0f ), vec3( 31, 63, 31 ) );
		newMax16 = newMaxVal.x * 2048.0f + newMaxVal.y * 32.0f + newMaxVal.z;

		const vec3 newMinVal = clamp( floor( ( at2 * xx - at1 * xy ) * f_rb_g.xyx + 0.5f ), vec3( 0.0f, 0.0f, 0.0f ), vec3( 31, 63, 31 ) );
		newMin16 = newMinVal.x * 2048.0f + newMinVal.y * 32.0f + newMinVal.z;
	}

	inOutMinEndp16 = newMin16;
	inOutMaxEndp16 = newMax16;

	return oldMin != newMin16 || oldMax != newMax16;
}

// srcPixelsBlock holds the 16 texels row by row, packed with packUnorm4x8 with alpha set to 1. refinements is the number
// of least squares passes (Betsy defaults to 2). Returns the 8 bytes of the block.
uvec2 bc1_compress_block( const uint srcPixelsBlock[ 16 ], uint refinements )
{
	bool bAllColoursEqual = true;
	for ( uint i = 1u; i < 16u; ++i )
	{
		bAllColoursEqual = bAllColoursEqual && srcPixelsBlock[ 0 ] == srcPixelsBlock[ i ];
	}

	float maxEndp16, minEndp16;
	uint mask = 0u;

	if ( bAllColoursEqual )
	{
		const uvec3 rgbVal = uvec3( round( unpackUnorm4x8( srcPixelsBlock[ 0 ] ).xyz * 255.0f ) );
		mask = 0xAAAAAAAAu;
		const vec2 endpoints = bc1_single_color_endpoints( rgbVal );
		maxEndp16 = endpoints.x;
		minEndp16 = endpoints.y;
	}
	else
	{
		// first step: pca+map along principal axis
		OptimizeColorsBlock( srcPixelsBlock, minEndp16, maxEndp16 );
		if ( minEndp16 != maxEndp16 )
		{
			vec3 colours[ 4 ];
			EvalColors( colours, maxEndp16, minEndp16 ); // Note min/max are inverted
			mask = MatchColorsBlock( srcPixelsBlock, colours );
		}

		// second step: refine (multiple times if requested)
		bool bStopRefinement = false;
		for ( uint i = 0u; i < refinements && !bStopRefinement; ++i )
		{
			const uint lastMask = mask;

			if ( RefineBlock( srcPixelsBlock, mask, minEndp16, maxEndp16 ) )
			{
				if ( minEndp16 != maxEndp16 )
				{
					vec3 colours[ 4 ];
					EvalColors( colours, maxEndp16, minEndp16 ); // Note min/max are inverted
					mask = MatchColorsBlock( srcPixelsBlock, colours );
				}
				else
				{
					mask = 0u;
					bStopRefinement = true;
				}
			}

			bStopRefinement = mask == lastMask || bStopRefinement;
		}
	}

	// write the color block
	if ( maxEndp16 < minEndp16 )
	{
		const float tmpValue = minEndp16;
		minEndp16 = maxEndp16;
		maxEndp16 = tmpValue;
		mask ^= 0x55555555u;
	}

	return uvec2( uint( maxEndp16 ) | ( uint( minEndp16 ) << 16u ), mask );
}
