#pragma once

#include "StratusCommon.h"

// For forcing 1-byte tight struct packing (we need to precisely control alignment and padding)
#ifdef __GNUC__
#define PACKED_STRUCT_ATTRIBUTE __attribute__ ((packed))
#else
#define PACKED_STRUCT_ATTRIBUTE
#endif

// Synchronized with definitions in pbr.glsl
#define MAX_TOTAL_SHADOW_ATLASES (5)
#define MAX_TOTAL_SHADOWS_PER_ATLAS (300)
#define MAX_TOTAL_SHADOW_MAPS (MAX_TOTAL_SHADOW_ATLASES * MAX_TOTAL_SHADOWS_PER_ATLAS)

// Once a VPL is further than this distance away it is automatically culled
#define MAX_VPL_DISTANCE_TO_VIEWER (500.0f)
#define MAX_TOTAL_VPL_SHADOW_MAPS (1024)

// Matches the definitions in common.glsl
#define GPU_DIFFUSE_MAPPED            (BITMASK_POW2(1))
#define GPU_AMBIENT_MAPPED            (BITMASK_POW2(2))
#define GPU_NORMAL_MAPPED             (BITMASK_POW2(3))
#define GPU_DEPTH_MAPPED              (BITMASK_POW2(4))
#define GPU_ROUGHNESS_MAPPED          (BITMASK_POW2(5))
#define GPU_METALLIC_MAPPED           (BITMASK_POW2(6))
// It's possible to have metallic + roughness combined into a single map
#define GPU_METALLIC_ROUGHNESS_MAPPED (BITMASK_POW2(7))

// Matches the definitions in vpl_common.glsl
#define MAX_TOTAL_VPLS_BEFORE_CULLING (4096)
#define MAX_TOTAL_VPLS_PER_FRAME (1024)
#define MAX_VPLS_PER_TILE (12)

namespace stratus {
    // Used with bindless textures
    typedef uint64_t GpuTextureHandle;

    // Matches the definition in vpl_common.glsl
    // See https://fvcaputo.github.io/2019/02/06/memory-alignment.html for alignment info
#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE alignas(16) GpuVec {
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

        glm::vec4 ToVec4() const {
            return glm::vec4(v[0], v[1], v[2], v[3]);
        }

    private:
        void _Copy(const glm::vec4& vec) {
            v[0] = vec.x;
            v[1] = vec.y;
            v[2] = vec.z;
            v[3] = vec.w;
        }
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

    // Matches the definition in common.glsl and is intended to be used with std430 layout
    // qualifier (SSBO and not UBO since UBO does not support std430).
#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuMaterial {
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
        unsigned int placeholder1_;

        GpuMaterial() {}
        GpuMaterial(const GpuMaterial&) = default;
        GpuMaterial(GpuMaterial&&) = default;

        GpuMaterial& operator=(const GpuMaterial&) = default;
        GpuMaterial& operator=(GpuMaterial&&) = default;
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuMeshData {
        float position[3];
        float texCoord[2];
        float normal[3];
        float tangent[3];
        float bitangent[3];
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

    // See "Drawing Commands" in OpenGL SuperBible and 
    // "Implementing Lightweight Rendering Queues" in 3D Graphics Rendering Cookbook
#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuDrawElementsIndirectCommand {
        uint32_t vertexCount;
        uint32_t instanceCount;
        // Measured in units of indices instead of the normal bytes
        uint32_t firstIndex;
        int32_t baseVertex;
        uint32_t baseInstance;
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuVplStage1PerTileOutputs {
        GpuVec averageLocalPosition;
        GpuVec averageLocalNormal;
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuVplStage2PerTileOutputs {
        int numVisible;
        int indices[MAX_VPLS_PER_TILE];

        GpuVplStage2PerTileOutputs() :
            numVisible(0) {}
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuVplData {
        GpuVec position;
        GpuVec color;
        GpuVec placeholder1_;
        float radius;
        float farPlane;
        float intensity;
        float placeholder2_;

        GpuVplData() :
            position(0.0f),
            color(0.0f),
            radius(1.0f),
            farPlane(1.0f),
            intensity(1.0f) {}
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    // Axis-Aligned Bounding Box from aabb.glsl
    struct PACKED_STRUCT_ATTRIBUTE GpuAABB {
        GpuVec vmin;
        GpuVec vmax;
        //GpuVec center;
        //GpuVec size;

        GpuAABB() :
            vmin(0.0f),
            vmax(1.0f) {}
            //center(0.0f),
            //size(1.0f) {}
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

    // This is synchronized with the version inside of pbr.fs
#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuPointLight {
        GpuVec position;
        GpuVec color;
        float radius;
        float farPlane;
        float placeholder_[2];
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

    // This is synchronized with the version inside of pbr.glsl
#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuAtlasEntry {
        int index = -1;
        int layer = -1;
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

    // This is synchronized with the version inside of common.glsl
#ifndef __GNUC__
    #pragma pack(push, 1)
#endif
    struct PACKED_STRUCT_ATTRIBUTE GpuHaltonEntry {
        float base2;
        float base3;
    };
#ifndef __GNUC__
    #pragma pack(pop)
#endif

    // These are here since if they fail the engine will not work
    static_assert(sizeof(GpuVec) == 16);
    static_assert(sizeof(GpuMaterial) == 128);
    static_assert(sizeof(GpuMeshData) == 56);
    static_assert(sizeof(GpuVplStage1PerTileOutputs) == 32);
    static_assert(sizeof(GpuVplStage2PerTileOutputs) == 52);
    static_assert(sizeof(GpuVplData) == 64);
    static_assert(sizeof(GpuAABB) == 32);
    static_assert(sizeof(GpuPointLight) == 48);
    static_assert(sizeof(GpuAtlasEntry) == 8);
    static_assert(sizeof(GpuHaltonEntry) == 8);
    static_assert(MAX_TOTAL_VPLS_PER_FRAME > 64);
}