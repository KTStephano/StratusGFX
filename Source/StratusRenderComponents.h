#pragma once

#include <vector>
#include <memory>
#include "StratusCommon.h"
#include "StratusGpuBuffer.h"
#include "StratusMaterial.h"
#include "StratusMath.h"
#include "StratusEntity.h"
#include "StratusEntityCommon.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "StratusRenderNode.h"

namespace stratus {
    // enum class RenderFaceCulling : int {
    //     CULLING_NONE,
    //     CULLING_CW,     // Clock-wise
    //     CULLING_CCW,    // Counter-clock-wise
    // };

    struct Mesh;

    typedef Mesh * MeshPtr;

    extern EntityPtr CreateRenderEntity();
    extern void InitializeRenderEntity(const EntityPtr&);

    struct Mesh final {
    private:
        Mesh();

    private:
        static Mesh * _PlacementNew(uint8_t *);

    public:
        static MeshPtr Create();
        static void Destroy(MeshPtr);

    public:
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

        // Before data has been moved to the GPU the Mesh will need to pack
        // all the data into a single buffer. This function is exposed so the
        // resource manager can do this asynchronously before moving to the graphics
        // application thread.
        void PackCpuData();

        // Temporary - to be removed
        void Render(size_t numInstances, const GpuArrayBuffer& additionalBuffers) const;

    private:
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
            bool needsRepacking = false;
        };

    private:
        GpuArrayBuffer _buffers;
        _MeshCpuData * _cpuData;
        size_t _dataSizeBytes;
        uint32_t _numVertices;
        uint32_t _numIndices;
        RenderFaceCulling _cullMode = RenderFaceCulling::CULLING_CCW;
    };

    struct MeshData {
        std::vector<MeshPtr> meshes;
        std::vector<glm::mat4> transforms;

        ~MeshData() {
            for (auto ptr : meshes) {
                Mesh::Destroy(ptr);
            }
        }
    };

    ENTITY_COMPONENT_STRUCT(RenderComponent)
        // Mesh data is always shared between components - changing one
        // changes all the RenderComponents that rely on it
        std::shared_ptr<MeshData> meshes;

        RenderComponent();
        RenderComponent(const RenderComponent&);

        MeshPtr GetMesh(const size_t) const;
        const glm::mat4& GetMeshTransform(const size_t) const;
        size_t GetMeshCount() const;

        // There will always be 1 material per mesh
        size_t GetMaterialCount() const;
        const std::vector<MaterialPtr>& GetAllMaterials() const;
        const MaterialPtr& GetMaterialAt(size_t) const;
        void AddMaterial(MaterialPtr);
        void SetMaterialAt(MaterialPtr, size_t);

    private:
        // This is per RenderComponent which means the same mesh may end up being
        // used with multiple different materials
        std::vector<MaterialPtr> _materials;
    };

    // If enabled then the entity interacts with light, otherwise it is flat shaded
    ENTITY_COMPONENT_STRUCT(LightInteractionComponent)
        LightInteractionComponent() = default;
        LightInteractionComponent(const LightInteractionComponent&) = default;
    };

    // If enabled then changes to position, orientation and scale are not tracked by renderer
    ENTITY_COMPONENT_STRUCT(StaticObjectComponent)
        StaticObjectComponent() = default;
        StaticObjectComponent(const StaticObjectComponent&) = default;
    };
}