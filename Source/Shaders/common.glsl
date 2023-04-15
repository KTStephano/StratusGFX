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
#define GPU_EMISSIVE_MAPPED           (BITMASK_POW2(2))
#define GPU_NORMAL_MAPPED             (BITMASK_POW2(3))
#define GPU_DEPTH_MAPPED              (BITMASK_POW2(4))
#define GPU_ROUGHNESS_MAPPED          (BITMASK_POW2(5))
#define GPU_METALLIC_MAPPED           (BITMASK_POW2(6))
// It's possible to have metallic + roughness combined into a single map
#define GPU_METALLIC_ROUGHNESS_MAPPED (BITMASK_POW2(7))

#define FLOAT2_TO_VEC2(f2) vec2(f2[0], f2[1])
#define FLOAT3_TO_VEC3(f3) vec3(f3[0], f3[1], f3[2])
#define FLOAT3_TO_VEC4(f3) vec4(FLOAT3_TO_VEC3(f3), 1.0)
#define FLOAT4_TO_VEC4(f4) vec4(f4[0], f4[1], f4[2], f4[3])

// Matches the definition in StratusGpuCommon.h
struct Material {
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D diffuseMap;
    sampler2D emissiveMap;
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D normalMap;
    sampler2D depthMap;
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D roughnessMap;
    sampler2D metallicMap;
    // total bytes next 3 entries = vec4 (for std430)
    sampler2D metallicRoughnessMap;
    float diffuseColor[4];
    float emissiveColor[3];
    // Base and max are interpolated between based on metallic
    // metallic of 0 = base reflectivity
    // metallic of 1 = max reflectivity
    float baseReflectivity[3];
    float maxReflectivity[3];
    // First two values = metallic, roughness
    float metallicRoughness[2];
    uint flags;
};

struct DrawElementsIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    // Measured in units of indices instead of the normal bytes
    uint firstIndex;
    int baseVertex;
    uint baseInstance;
};

struct HaltonEntry {
    float base2;
    float base3;
};

layout (std430, binding = 30) readonly buffer SSBO_Global1 {
    Material materials[];
};

layout (std430, binding = 31) readonly buffer SSBO_Global2 {
    uint materialIndices[];
};

// Checks if flag & mask is greater than 0
bool bitwiseAndBool(uint flag, uint mask) {
    uint value = flag & mask;
    return value != 0;
}

// Prevents HDR color values from exceeding 16-bit color buffer range
vec3 boundHDR(vec3 value) {
    return min(value, 65504.0);
    //return value; // Engine is currently using 32-bit... disable for now
}

// See https://community.khronos.org/t/saturate/53155
vec3 saturate(vec3 value) {
    return clamp(value, 0.0, 1.0);
}

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec3 clampMediumPrecision(vec3 value) {
    return clamp(value, 0.0, 65504.0);
}

float clampMediumPrecision(float value) {
    return clamp(value, 0.0, 65504.0);
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

// See https://stackoverflow.com/questions/32227283/getting-world-position-from-depth-buffer-value
vec3 worldPositionFromDepth(in vec2 uv, in float depth, in mat4 invProjectionView) {
    // Convert depth from [0, 1] to [-1, 1]
    float z = depth * 2.0 - 1.0;

    // Set up NDC using -1, 1 tex coords and -1, 1 z coordinate
    vec4 ndc = vec4(uv * 2.0 - 1.0, z, 1.0);

    // Convert to world space
    vec4 worldPosition = invProjectionView * ndc;

    // Perform perspective divide to complete the transform
    return worldPosition.xyz / worldPosition.w;
}

// See https://sugulee.wordpress.com/2021/06/21/temporal-anti-aliasingtaa-tutorial/
vec2 calculateVelocity(in vec4 currentClipPos, in vec4 prevClipPos) {
    // Perform perspective divide
    vec2 current = currentClipPos.xy / currentClipPos.w;
    vec2 prev = prevClipPos.xy / prevClipPos.w;

    // Move from clip space [-1, 1] to uv space [0, 1]
    //current = current * 0.5 + 0.5;
    //prev = prev * 0.5 + 0.5;

    return (current - prev).xy * 0.5;
}

// See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }