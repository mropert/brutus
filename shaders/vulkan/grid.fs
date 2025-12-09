#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) in vec2 inGridPos;
layout (location = 1) in vec3 inViewPos;
layout (location = 0) out vec4 outFragColor;

const float NearDistance = 10;
const float FarDistance = 20;
const int GridMinorsPerMajor = 4;
const vec3 KeyColor[2] = { { 1, 0, 0 }, { 0, 0, 1 } };

float Fade(float value)
{
	return 1 - smoothstep(NearDistance, FarDistance, value);
}

float ScaledFade(float value, vec2 fogScale)
{
	return 1 - smoothstep(NearDistance * fogScale.x, FarDistance * fogScale.y, value);
}

void main()
{
	// figure out our fogging values
	float viewDist = length(inViewPos);
	float majorFog = Fade(viewDist);
	float minorFog = ScaledFade(viewDist, vec2(0.5, 0.85));

	// find the index of the closest grid line to this pixel
	ivec2 lineIndex = ivec2(floor(inGridPos));

	// pick an appropriate width and color for the closest line (in *each* of X and Y!)
	vec2 lineWidth;
	vec3 lineColor[2];
	vec2 lineFog;
	for (int i = 0; i < 2; i++)
	{
		float width;
		vec3 color;
		float fog;

		if (lineIndex[i] == 0)
		{
			width = 5;
			color = KeyColor[1 - i];
			fog = majorFog;
		}
		else if (lineIndex[i] % GridMinorsPerMajor == 0)
		{
			width = 3;
			color = vec3(0.5);
			fog = majorFog;
		}
		else
		{
			width = 2;
			color = vec3(0.25);
			fog = minorFog;
		}

		lineWidth[i] = width;
		lineColor[i] = color;
		lineFog[i] = fog;
	}

	vec2 lineDist = abs(0.5 - fract(inGridPos));
	vec2 lineMask = 1 - clamp(lineDist / (fwidth(inGridPos) * lineWidth), 0, 1);

	vec2 blendFactors = lineMask * lineFog;
	for (int i = 0; i < 2; i++)
		if (lineIndex[1 - i] == 0 && lineIndex[i] != 0)
			blendFactors[i] *= smoothstep(0, 0.5, lineDist[1 - i]);

    float alpha = max(blendFactors.x, blendFactors.y);
	vec3 color = max(lineColor[0] * blendFactors.x, lineColor[1] * blendFactors.y);
    outFragColor = vec4( mix( vec3( 0.1 ), color, alpha), 1 );
}
