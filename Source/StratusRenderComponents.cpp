#pragma once

#include <vector>
#include <memory>
#include "StratusCommon.h"
#include "StratusGpuBuffer.h"
#include "StratusMaterial.h"
#include "StratusMath.h"
#include "StratusEntity2.h"
#include "StratusEntityCommon.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace stratus {
    enum class RenderFaceCulling : int {
        CULLING_NONE,
        CULLING_CW,     // Clock-wise
        CULLING_CCW,    // Counter-clock-wise
    };

    struct Mesh;

    typedef std::shared_ptr<Mesh> MeshPtr;

    extern Entity2Ptr CreateRenderEntity();
    extern void InitializeRenderEntity(const Entity2Ptr&);

    struct Mesh final {
        Mesh();
        ~Mesh();

        void AddVertex(const glm::vec3&);
        void AddUV(const glm::vec2&);
        void AddNormal(const glm::vec3&);
        void AddTangent(const glm::vec3&);
        void AddBitangent(const glm::vec3&);
        void AddIndex(uint32_t);

        bool IsFinalized() const;
        void FinalizeData();

        const GpuArrayBuffer& GetData() const;
        size_t GetGpuSizeBytes() const;

        void SetFaceCulling(const RenderFaceCulling&);
        RenderFaceCulling GetFaceCulling() const;

    private:
        void _GenerateCpuData();
        void _GenerateGpuData();
        void _CalculateTangentsBitangents();
        void _EnsureFinalized() const;
        void _EnsureNotFinalized() const;

    private:
        struct _MeshCpuData {
            std::vector<glm::vec3> vertices;
            std::vector<glm::vec2> uvs;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec3> tangents;
            std::vector<glm::vec3> bitangents;
            std::vector<uint32_t> indices;
            std::vector<float> data;
        };

    private:
        GpuArrayBuffer _buffers;
        _MeshCpuData * _cpuData;
        size_t _dataSizeBytes;
        uint32_t _numVertices;
        uint32_t _numIndices;
        RenderFaceCulling _cullMode = RenderFaceCulling::CULLING_CCW;
    };

    ENTITY_COMPONENT_STRUCT(RenderComponent) {
        std::vector<MeshPtr> meshes;
        std::vector<MaterialPtr> materials;
        std::vector<glm::mat4> transforms;
    };

    ENTITY_COMPONENT_STRUCT(LightInteractionComponent)
    };

    ENTITY_COMPONENT_STRUCT(StaticObjectComponent)
    };
}