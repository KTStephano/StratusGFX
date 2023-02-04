#pragma once

#include "glm/glm.hpp"

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
    };
}