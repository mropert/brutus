// View depth range of the visible pixels, measured by depth_range.comp for the next frame's shadow cascades. Sky pixels
// (depth 0) are left out, nearest is 0 when nothing was rendered.
// The raw depths are stored as uints on purpose: float atomic min/max are optional (VK_EXT_shader_atomic_float2), but
// non-negative IEEE floats order exactly like their bits read as uints (sign bit 0, then exponent, then mantissa). Reverse-Z
// depths are in [0, 1], so integer atomics on floatBitsToUint() give the float min/max. Read with uintBitsToFloat()
layout( buffer_reference, std430 ) buffer DepthRange
{
	uint nearest; // Largest depth
	uint farthest; // Smallest depth
};
