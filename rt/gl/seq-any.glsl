include(preamble.glsl)
include(tri-is.glsl)

// assumes 32 bit
#define FLT_MAX 3.402823466e+38
uniform int N;
void run(uint x, uint y) {
	uint id = y * w + x;
	vec4 closest = vec4(FLT_MAX, -1, -1, 0), is;
	vec4 o = rays[id],
	d = rays[w*h + id];
	for (int i = 0; i < N; ++i)
		if (intersect(i, o, d, vec2(0,FLT_MAX), is)) {
			is.w = intBitsToFloat(i);
			intersections[id] = is;
			break;
		}
}

