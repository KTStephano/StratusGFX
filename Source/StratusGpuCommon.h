#pragma once

#include "StratusCommon.h"

// Matches the definitions in common.glsl
#define GPU_DIFFUSE_MAPPED            (BITMASK_POW2(1))
#define GPU_AMBIENT_MAPPED            (BITMASK_POW2(2))
#define GPU_NORMAL_MAPPED             (BITMASK_POW2(3))
#define GPU_DEPTH_MAPPED              (BITMASK_POW2(4))
#define GPU_ROUGHNESS_MAPPED          (BITMASK_POW2(5))
#define GPU_METALLIC_MAPPED           (BITMASK_POW2(6))
// It's possible to have metallic + roughness combined into a single map
#define GPU_METALLIC_ROUGHNESS_MAPPED (BITMASK_POW2(7))

namespace stratus {
    // Used with bindless textures
    typedef uint64_t GpuTextureHandle;

    // Matches the definition in vpl_tiled_deferred_culling.glsl
    // See https://fvcaputo.github.io/2019/02/06/memory-alignment.html for alignment info
    struct alignas(16) GpuVec {
        float v[4];

        GpuVec(float x, float y, float z, float w) {
            v[0] = x;
            v[1] = y;
            v[2] = z;
            v[3] = w;
        }

        GpuVec(float xyzw) : GpuVec(xyzw, xyzw, xyzw, xyzw) {}
        GpuVec(const glm::vec4& v) : GpuVec(v[0], v[1], v[2], v[3]) {}
        GpuVec(const glm::vec3& v) : GpuVec(glm::vec4(v, 0.0f)) {}
        GpuVec() : GpuVec(0.0f) {}

        GpuVec& operator=(const GpuVec&) = default;
        GpuVec& operator=(GpuVec&&) = default;

        GpuVec& operator=(const glm::vec4& vec) {
            v[0] = vec.x;
            v[1] = vec.y;
            v[2] = vec.z;
            v[3] = vec.w;
            return *this;
        }

        GpuVec& operator=(const glm::vec3& vec) {
            return this->operator=(glm::vec4(vec, 0.0f));
        }

        GpuVec& operator=(float xyzw) {
            return this->operator=(glm::vec4(xyzw, xyzw, xyzw, xyzw));
        }
    };

    // Matches the definition in common.glsl and is intended to be used with std430 layout
    // qualifier (SSBO and not UBO since UBO does not support std430).
    struct GpuMaterial {
        GpuVec diffuseColor;
        GpuVec ambientColor;
        GpuVec baseReflectivity;
        // First two values = metallic, roughness
        // last two values = padding
        GpuVec metallicRoughness;
        // total bytes next 2 entries = GpuVec
        alignas(8) GpuTextureHandle diffuseMap;
        alignas(8) GpuTextureHandle ambientMap;
        // total bytes next 2 entries = GpuVec
        alignas(8) GpuTextureHandle normalMap;
        alignas(8) GpuTextureHandle depthMap;
        // total bytes next 2 entries = GpuVec
        alignas(8) GpuTextureHandle roughnessMap;
        alignas(8) GpuTextureHandle metallicMap;
        // total bytes next 2 entries = GpuVec
        alignas(8) GpuTextureHandle metallicRoughnessMap;
        alignas(4) unsigned int flags = 0;
        alignas(4) unsigned int _1;
    };
}