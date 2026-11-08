// BC4 block encoder (UNORM). Include only: no bindings, no main.
//
// From Betsy (https://github.com/darksylinc/betsy), commit cc723dc, bin/Data/bc4.glsl.
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
// Changes from Betsy: a function per block instead of a shader with its own bindings and main, one thread per block
// instead of 4 threads with shared memory and atomics, values rounded to integers (the bias below compares them exactly),
// no SNORM.

// values holds the block's 16 texels row by row, in [0, 1]. Returns the 8 bytes of the block.
uvec2 bc4_compress_block( float values[ 16 ] )
{
	float minVal = 255.0f;
	float maxVal = 0.0f;
	for ( uint i = 0u; i < 16u; ++i )
	{
		values[ i ] = round( values[ i ] * 255.0f );
		minVal = min( minVal, values[ i ] );
		maxVal = max( maxVal, values[ i ] );
	}

	// determine bias and emit color indices
	// given the choice of maxVal/minVal, these indices are optimal:
	// http://fgiesen.wordpress.com/2009/12/15/dxt5-alpha-block-index-determination/
	float dist = maxVal - minVal;
	float dist4 = dist * 4.0f;
	float dist2 = dist * 2.0f;
	float bias = ( dist < 8 ) ? ( dist - 1 ) : ( trunc( dist * 0.5f ) + 2 );
	bias -= minVal * 7;

	uint mask0 = 0u, mask1 = 0u;

	for ( uint i = 0u; i < 16u; ++i )
	{
		float a = values[ i ] * 7.0f + bias;

		int ind = 0;

		// select index. this is a "linear scale" lerp factor between 0 (val=min) and 7 (val=max).
		if ( a >= dist4 )
		{
			ind = 4;
			a -= dist4;
		}

		if ( a >= dist2 )
		{
			ind += 2;
			a -= dist2;
		}

		if ( a >= dist )
			ind += 1;

		// turn linear scale into DXT index (0/1 are extremal pts)
		ind = -ind & 7;
		ind ^= ( 2 > ind ) ? 1 : 0;

		// write index
		const uint bits = 16u + i * 3u;
		if ( bits < 32u )
		{
			mask0 |= uint( ind ) << bits;
			if ( bits + 3u > 32u )
			{
				mask1 |= uint( ind ) >> ( 32u - bits );
			}
		}
		else
		{
			mask1 |= uint( ind ) << ( bits - 32u );
		}
	}

	return uvec2( uint( maxVal ) | ( uint( minVal ) << 8u ) | mask0, mask1 );
}
