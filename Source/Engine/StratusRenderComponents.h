#pragma once

#include <vector>
#include <list>
#include <memory>
#include "StratusCommon.h"
#include "StratusGpuBuffer.h"
#include "StratusGpuCommon.h"
#include "StratusMaterial.h"
#include "StratusMath.h"
#include "StratusEntity.h"
#include "StratusEntityCommon.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "StratusTypes.h"

#define GPU_MESH_CHUNK_SIZE (65536)

namespace stratus {
    enum class RenderFaceCulling : int {
        CULLING_NONE,
        CULLING_CW,     // Clock-wise
        CULLING_CCW,    // Counter-clock-wise
    };

    struct Mesh;
    struct Meshlet;

    typedef Mesh* MeshPtr;
    typedef Meshlet* MeshletPtr;

    extern EntityPtr CreateRenderEntity();
    extern void InitializeRenderEntity(const EntityPtr&);

    struct Meshlet final {
    private:
        static Meshlet* PlacementNew_(u8*);

    public:
        static MeshletPtr Create();
        static void Destroy(MeshletPtr);

        Meshlet();
        ~Meshlet();

        void AddVertex(const glm::vec3&);
        void AddUV(const glm::vec2&);
        void AddNormal(const glm::vec3&);
        void AddTangent(const glm::vec3&);
        void AddBitangent(const glm::vec3&);
        void AddIndex(u32);

        void ReserveVertices(usize);
        void ReserveIndices(usize);

        bool IsFinalized() const;
        void FinalizeData();

        usize GetGpuSizeBytes() const;

        // Before data has been moved to the GPU the Mesh will need to pack
        // all the data into a single buffer. This function is exposed so the
        // resource manager can do this asynchronously before moving to the graphics
        // application thread.
        void PackCpuData();
        void CalculateAabbs(const glm::mat4& transform);
        void GenerateLODs();

        // Temporary - to be removed
        void Render(usize numInstances, const GpuArrayBuffer& additionalBuffers) const;

        // Offsets into global GPU buffers
        u32 GetVertexOffset() const;
        u32 GetIndexOffset(usize lod) const;
        u32 GetNumIndices(usize lod) const;
        // Returns all vertices from all LODs
        u32 GetTotalNumVertices() const;
        // Returns all indices from all LODs
        u32 GetTotalNumIndices() const;

        const GpuAABB& GetAABB() const;

    private:
        void GenerateGpuData_();
        void CalculateTangentsBitangents_();
        void EnsureFinalized_() const;
        void EnsureNotFinalized_() const;

    private:
        struct MeshCpuData_ {
            std::vector<glm::vec3> vertices;
            std::vector<glm::vec2> uvs;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec3> tangents;
            std::vector<glm::vec3> bitangents;
            std::vector<u32> indices;
            std::vector<GpuMeshData> data;
            std::vector<std::vector<u32>> indicesPerLod;
            bool needsRepacking = false;
        };

    private:
        MeshCpuData_* cpuData_;
        GpuAABB aabb_;
        usize dataSizeBytes_;
        u32 numVertices_;
        u32 numIndices_;
        u32 vertexOffset_; // Into global GpuBuffer
        std::vector<u32> numIndicesPerLod_;
        std::vector<u32> indexOffsetPerLod_; // Into global GpuBuffer
        u32 numIndicesApproximateLod_;
    };

    struct Mesh final {
    private:
        Mesh();

    private:
        static Mesh* PlacementNew_(u8*);

    public:
        static MeshPtr Create();
        static void Destroy(MeshPtr);

        //bool IsFinalized() const;

        MeshletPtr NewMeshlet();

        const MeshletPtr GetMeshlet(const usize) const;
        MeshletPtr GetMeshlet(const usize);

        usize NumMeshlets() const;

        void SetFaceCulling(const RenderFaceCulling&);
        RenderFaceCulling GetFaceCulling() const;

    public:
        ~Mesh();

    private:
        std::vector<MeshletPtr> meshlets_;
        RenderFaceCulling cullMode_ = RenderFaceCulling::CULLING_CCW;
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

    MeshPtr GetMesh(const usize) const;
    const glm::mat4& GetMeshTransform(const usize) const;
    usize GetMeshCount() const;

    // There will always be 1 material per mesh
    usize GetMaterialCount() const;
    const std::vector<MaterialPtr>& GetAllMaterials() const;
    const MaterialPtr& GetMaterialAt(usize) const;
    void AddMaterial(MaterialPtr);
    void SetMaterialAt(MaterialPtr, usize);

private:
    // This is per RenderComponent which means the same mesh may end up being
    // used with multiple different materials
    std::vector<MaterialPtr> materials_;
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