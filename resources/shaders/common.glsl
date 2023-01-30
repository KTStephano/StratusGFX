STRATUS_GLSL_VERSION

#define PI 3.14159265359
#define PREVENT_DIV_BY_ZERO 0.00001
// See https://stackoverflow.com/questions/16069959/glsl-how-to-ensure-largest-possible-float-value-without-overflow
#define FLOAT_MAX 3.402823466e+38
#define FLOAT_MIN 1.175494351e-38
#define DOUBLE_MAX 1.7976931348623158e+308
#define DOUBLE_MIN 2.2250738585072014e-308

// Prevents HDR color values from exceeding 16-bit color buffer range
vec3 boundHDR(vec3 value) {
    return min(value, 65500.0);
}

// See https://community.khronos.org/t/saturate/53155
vec3 saturate(vec3 value) {
    return clamp(value, 0.0, 1.0);
}

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec2 computeTexelWidth(sampler2D tex, int miplevel) {
    // This will give us the size of a single texel in (x, y) directions
    // (miplevel is telling it to give us the size at mipmap *miplevel*, where 0 would mean full size image)
    return (1.0 / textureSize(tex, miplevel));// * vec2(2.0, 1.0);
}

vec2 computeTexelWidth(sampler2DArrayShadow tex, int miplevel) {
    // This will give us the size of a single texel in (x, y) directions
    // (miplevel is telling it to give us the size at mipmap *miplevel*, where 0 would mean full size image)
    return (1.0 / textureSize(tex, miplevel).xy);// * vec2(2.0, 1.0);
}

// Linear interpolate
vec4 lerp(vec4 x, vec4 y, float a) {
    return mix(x, y, a);
}

vec3 lerp(vec3 x, vec3 y, float a) {
    return mix(x, y, a);
}

float lerp(float x, float y, float a) {
    return mix(x, y, a);
}