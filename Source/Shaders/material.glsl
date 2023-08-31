STRATUS_GLSL_VERSION

#include "bindings.glsl"

// Matches the definition in StratusGpuCommon.h
struct Material {
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D diffuseMap;
    sampler2D emissiveMap;
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D normalMap;
    //sampler2D depthMap;
    // total bytes next 2 entries = vec4 (for std430)
    sampler2D roughnessMap;
    sampler2D metallicMap;
    // total bytes next 3 entries = vec4 (for std430)
    sampler2D metallicRoughnessMap;
    // 4x8 bits fixed point
    uint diffuseColor;
    // 3x8 bits fixed point, 1x8 bits unused
    uint emissiveColor;
    // Base and max are interpolated between based on metallic
    // metallic of 0 = base reflectivity
    // metallic of 1 = max reflectivity
    //
    // 1x8 bits reflectance, 1x8 bits metallic, 1x8 bits roughness
    uint reflectanceMetallicRoughness;
    uint flags;
};

layout (std430, binding = MATERIAL_BINDING_POINT) readonly buffer SSBO_Global1 {
    Material materials[];
};

layout (std430, binding = MATERIAL_INDICES_BINDING_POINT) readonly buffer SSBO_Global2 {
    uint materialIndices[];
};

vec4 decodeMaterialData(in uint data) {
    const float invMax = 1.0 / 255.0;

    vec4 result = vec4(0.0);

    result.r = float((data & 0xFF000000) >> 24) * invMax;
    result.g = float((data & 0x00FF0000) >> 16) * invMax;
    result.b = float((data & 0x0000FF00) >>  8) * invMax;
    result.a = float((data & 0x000000FF)      ) * invMax;

    return result;
}