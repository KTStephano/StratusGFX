#pragma once

#include <memory>
#include "StratusGpuBuffer.h"
#include <vector>
#include "StratusEntity.h"
#include "StratusRenderComponents.h"
#include <unordered_map>
#include <unordered_set>
#include "StratusGpuMaterialBuffer.h"
#include "StratusTransformComponent.h"

namespace stratus {
    struct GpuCommandBuffer2;
    struct GpuCommandManager;

    typedef std::shared_ptr<GpuCommandBuffer2> GpuCommandBuffer2Ptr;
    typedef std::shared_ptr<GpuCommandManager> GpuCommandManagerPtr;

    // Stores material indices, model transforms and indirect draw commands
    struct GpuCommandBuffer2 final {
        GpuCommandBuffer2(const RenderFaceCulling&, size_t numLods, size_t commandBlockSize);

        GpuCommandBuffer2(GpuCommandBuffer2&&) = default;
        GpuCommandBuffer2(const GpuCommandBuffer2&) = delete;

        GpuCommandBuffer2& operator=(GpuCommandBuffer2&&) = delete;
        GpuCommandBuffer2& operator=(const GpuCommandBuffer2&) = delete;

        size_t NumDrawCommands() const;
        size_t NumLods() const;
        const RenderFaceCulling& GetFaceCulling() const;

        void RecordCommand(const RenderComponent*, const MeshWorldTransforms*, const size_t, const size_t);
        void RemoveAllCommands(const RenderComponent*);

        void UpdateTransforms(const RenderComponent*, const MeshWorldTransforms*);
        void UpdateMaterials(const RenderComponent*, const GpuMaterialBufferPtr&);

        void UploadDataToGpu();

        void BindMaterialIndicesBuffer(uint32_t index);
        void BindPrevFrameModelTransformBuffer(uint32_t index);
        void BindModelTransformBuffer(uint32_t index);
        void BindAabbBuffer(uint32_t index);

        void BindIndirectDrawCommands(const size_t lod);
        void UnbindIndirectDrawCommands(const size_t lod);

        const GpuBuffer& GetIndirectDrawCommandsBuffer(const size_t lod) const;
        const GpuBuffer& GetVisibleDrawCommandsBuffer() const;
        const GpuBuffer& GetSelectedLodDrawCommandsBuffer() const;

        static inline GpuCommandBuffer2Ptr Create(const RenderFaceCulling& cull, const size_t numLods, const size_t commandBlockSize) {
            return GpuCommandBuffer2Ptr(new GpuCommandBuffer2(cull, numLods, commandBlockSize));
        }

    private:
        bool InsertMeshPending_(const RenderComponent*, const MeshPtr);

    private:
        std::vector<GpuTypedBufferPtr<GpuDrawElementsIndirectCommand>> drawCommands_;
        GpuTypedBufferPtr<GpuDrawElementsIndirectCommand> visibleCommands_;
        GpuTypedBufferPtr<GpuDrawElementsIndirectCommand> selectedLodCommands_;
        GpuTypedBufferPtr<glm::mat4> prevFrameModelTransforms_;
        GpuTypedBufferPtr<glm::mat4> modelTransforms_;
        GpuTypedBufferPtr<GpuAABB> aabbs_;
        GpuTypedBufferPtr<uint32_t> materialIndices_;
        std::unordered_map<const RenderComponent *, std::unordered_map<const MeshPtr, uint32_t>> drawCommandIndices_;
        std::unordered_map<const RenderComponent *, std::unordered_set<const MeshPtr>> pendingMeshUpdates_;

        RenderFaceCulling culling_;
    };

    // Manages a set of command buffers
    struct GpuCommandManager {
        std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr> flatMeshes;
        std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr> dynamicPbrMeshes;
        std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr> staticPbrMeshes;

        GpuCommandManager(size_t numLods);

        size_t NumLods() const;

        void RecordCommands(const EntityPtr&, const GpuMaterialBufferPtr&);
        void RemoveAllCommands(const EntityPtr&);
        void ClearCommands();

        void UpdateTransforms(const EntityPtr&);
        void UpdateMaterials(const EntityPtr&, const GpuMaterialBufferPtr&);

        void UploadDataToGpu();

        static inline GpuCommandManagerPtr Create(const size_t numLods) {
            return GpuCommandManagerPtr(new GpuCommandManager(numLods));
        }

    private:
        std::unordered_set<EntityPtr> entities_;
        size_t numLods_;
    };
}