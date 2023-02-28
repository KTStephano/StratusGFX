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

// Matches the definitions in vpl_tiled_deferred_culling.glsl
#define MAX_TOTAL_VPLS_PER_FRAME (512)
#define MAX_VPLS_PER_TILE (6)

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
        
        GpuVec(const GpuVec& other) {
            _Copy(glm::vec4(other.v[0], other.v[1], other.v[2], other.v[3]));
        }

        GpuVec& operator=(const GpuVec& other) {
            _Copy(glm::vec4(other.v[0], other.v[1], other.v[2], other.v[3]));
            return *this;
        }

        GpuVec& operator=(GpuVec&& other) {
            _Copy(glm::vec4(other.v[0], other.v[1], other.v[2], other.v[3]));
            return *this;
        }

        GpuVec& operator=(const glm::vec4& vec) {
            _Copy(vec);
            return *this;
        }

        GpuVec& operator=(const glm::vec3& vec) {
            _Copy(glm::vec4(vec, 0.0f));
            return *this;
        }

        GpuVec& operator=(float xyzw) {
            _Copy(glm::vec4(xyzw, xyzw, xyzw, xyzw));
            return *this;
        }

    private:
        void _Copy(const glm::vec4& vec) {
            v[0] = vec.x;
            v[1] = vec.y;
            v[2] = vec.z;
            v[3] = vec.w;
        }
    };

    // Matches the definition in common.glsl and is intended to be used with std430 layout
    // qualifier (SSBO and not UBO since UBO does not support std430).
    struct alignas(128) GpuMaterial {
        GpuVec diffuseColor;
        GpuVec ambientColor;
        GpuVec baseReflectivity;
        // First two values = metallic, roughness
        // last two values = padding
        GpuVec metallicRoughness;
        // total bytes next 2 entries = GpuVec
        GpuTextureHandle diffuseMap;
        GpuTextureHandle ambientMap;
        // total bytes next 2 entries = GpuVec
        GpuTextureHandle normalMap;
        GpuTextureHandle depthMap;
        // total bytes next 2 entries = GpuVec
        // TODO: Remove these and always favor metallicRoughnessMap
        GpuTextureHandle roughnessMap;
        GpuTextureHandle metallicMap;
        // total bytes next 3 entries = GpuVec
        GpuTextureHandle metallicRoughnessMap;
        unsigned int flags = 0;
        unsigned int _1;

        GpuMaterial() {}
        GpuMaterial(const GpuMaterial&) = default;
        GpuMaterial(GpuMaterial&&) = default;

        GpuMaterial& operator=(const GpuMaterial&) = default;
        GpuMaterial& operator=(GpuMaterial&&) = default;
    };

    struct alignas(64) GpuMeshData {
        float position[3];
        float texCoord[2];
        float normal[3];
        float tangent[3];
        float bitangent[3];
        float _1[2];
    };

    // See "Drawing Commands" in OpenGL SuperBible and 
    // "Implementing Lightweight Rendering Queues" in 3D Graphics Rendering Cookbook
    struct GpuDrawElementsIndirectCommand {
        uint32_t vertexCount;
        uint32_t instanceCount;
        // Measured in units of indices instead of the normal bytes
        uint32_t firstIndex;
        int32_t baseVertex;
        uint32_t baseInstance;
    };

    struct alignas(32) GpuVplStage1PerTileOutputs {
        GpuVec averageLocalPosition;
        GpuVec averageLocalNormal;
    };

    struct alignas(32) GpuVplStage2PerTileOutputs {
        int numVisible;
        int _1;
        int indices[MAX_VPLS_PER_TILE];

        GpuVplStage2PerTileOutputs() :
            numVisible(0) {}
    };

    struct alignas(64) GpuVplData {
        GpuVec position;
        GpuVec color;
        GpuVec _3;
        float radius;
        float farPlane;
        float intensity;
        float  _2;

        GpuVplData() :
            position(0.0f),
            color(0.0f),
            radius(1.0f),
            farPlane(1.0f),
            intensity(1.0f) {}
    };

    static_assert(sizeof(GpuVec) == 16);
    static_assert(sizeof(GpuMaterial) == 128);
    static_assert(sizeof(GpuMeshData) == 64);
    static_assert(sizeof(GpuVplStage1PerTileOutputs) == 32);
    static_assert(sizeof(GpuVplStage2PerTileOutputs) == 32);
    static_assert(sizeof(GpuVplData) == 64);
}