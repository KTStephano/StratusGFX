STRATUS_GLSL_VERSION

#define PI 3.14159265359
#define PREVENT_DIV_BY_ZERO 0.00001
// See https://stackoverflow.com/questions/16069959/glsl-how-to-ensure-largest-possible-float-value-without-overflow
#define FLOAT_MAX 3.402823466e+38
#define FLOAT_MIN 1.175494351e-38
#define DOUBLE_MAX 1.7976931348623158e+308
#define DOUBLE_MIN 2.2250738585072014e-308
#define BITMASK_POW2(offset) (1 << offset)

// Matches the definitions in StratusGpuCommon.h
#define GPU_DIFFUSE_MAPPED            (BITMASK_POW2(1))
#define GPU_AMBIENT_MAPPED            (BITMASK_POW2(2))
#define GPU_NORMAL_MAPPED             (BITMASK_POW2(3))
#define GPU_DEPTH_MAPPED              (BITMASK_POW2(4))
#define GPU_ROUGHNESS_MAPPED          (BITMASK_POW2(5))
#define GPU_METALLIC_MAPPED           (BITMASK_POW2(6))
// It's possible to have metallic + roughness combined into a single map
#define GPU_METALLIC_ROUGHNESS_MAPPED (BITMASK_POW2(7))

// Matches the definition in StratusGpuCommon.h
struct Material {
    vec4 diffuseColor;
    vec4 ambientColor;
    vec4 baseReflectivity;
    // First two values = metallic, roughness
    // last two values = padding
    vec4 metallicRoughness;
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D diffuseMap;
    sampler2D ambientMap;
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D normalMap;
    sampler2D depthMap;
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D roughnessMap;
    sampler2D metallicMap;
    // total bytes next 3 entries = vec4 (for std430)
    sampler2D metallicRoughnessMap;
    uint flags;
    uint _1;
};

// Checks if flag & mask is greater than 0
bool bitwiseAndBool(uint flag, uint mask) {
    uint value = flag & mask;
    return value != 0;
}

// Prevents HDR color values from exceeding 16-bit color buffer range
vec3 boundHDR(vec3 value) {
    return min(value, 65500.0);
    //return value; // Engine is currently using 32-bit... disable for now
}

// See https://community.khronos.org/t/saturate/53155
vec3 saturate(vec3 value) {
    return clamp(value, 0.0, 1.0);
}

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec2 computeTexelSize(sampler2D tex, int miplevel) {
    // This will give us the size of a single texel in (x, y) directions
    // (miplevel is telling it to give us the size at mipmap *miplevel*, where 0 would mean full size image)
    return (1.0 / textureSize(tex, miplevel));// * vec2(2.0, 1.0);
}

vec2 computeTexelSize(sampler2DArrayShadow tex, int miplevel) {
    // This will give us the size of a single texel in (x, y) directions
    // (miplevel is telling it to give us the size at mipmap *miplevel*, where 0 would mean full size image)
    return (1.0 / textureSize(tex, miplevel).xy);// * vec2(2.0, 1.0);
}

vec2 computeTexelSize(samplerCube tex, int miplevel) {
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