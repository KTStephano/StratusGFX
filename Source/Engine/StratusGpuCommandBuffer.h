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
#include "StratusPointer.h"

namespace stratus {
    struct GpuCommandBuffer;
    struct GpuCommandManager;
    
    struct GpuCommandReceiveBuffer;
    struct GpuCommandReceiveManager;

    typedef UnsafePtr<GpuCommandBuffer> GpuCommandBufferPtr;
    typedef UnsafePtr<GpuCommandManager> GpuCommandManagerPtr;

    typedef UnsafePtr<GpuCommandReceiveBuffer> GpuCommandReceiveBufferPtr;
    typedef UnsafePtr<GpuCommandReceiveManager> GpuCommandReceiveManagerPtr;

    // Stores material indices, model transforms and indirect draw commands
    struct GpuCommandBuffer final {
        GpuCommandBuffer(const RenderFaceCulling&, size_t numLods, size_t commandBlockSize);

        GpuCommandBuffer(GpuCommandBuffer&&) = default;
        GpuCommandBuffer(const GpuCommandBuffer&) = delete;

        GpuCommandBuffer& operator=(GpuCommandBuffer&&) = delete;
        GpuCommandBuffer& operator=(const GpuCommandBuffer&) = delete;

        size_t NumDrawCommands() const;
        size_t NumLods() const;
        size_t CommandCapacity() const;
        const RenderFaceCulling& GetFaceCulling() const;

        void RecordCommand(RenderComponent*, MeshWorldTransforms*, const size_t, const size_t);
        void RemoveAllCommands(RenderComponent*);

        void UpdateTransforms(RenderComponent*, MeshWorldTransforms*);
        void UpdateMaterials(RenderComponent*, const GpuMaterialBufferPtr&);

        bool UploadDataToGpu();

        void BindMaterialIndicesBuffer(uint32_t index);
        void BindPrevFrameModelTransformBuffer(uint32_t index);
        void BindModelTransformBuffer(uint32_t index);
        void BindAabbBuffer(uint32_t index);

        void BindIndirectDrawCommands(const size_t lod);
        void UnbindIndirectDrawCommands(const size_t lod);

        GpuBuffer GetIndirectDrawCommandsBuffer(const size_t lod) const;
        GpuBuffer GetVisibleDrawCommandsBuffer() const;
        GpuBuffer GetSelectedLodDrawCommandsBuffer() const;

        static inline GpuCommandBufferPtr Create(const RenderFaceCulling& cull, const size_t numLods, const size_t commandBlockSize) {
            return GpuCommandBufferPtr(new GpuCommandBuffer(cull, numLods, commandBlockSize));
        }

    private:
        bool InsertMeshPending_(RenderComponent*, MeshPtr);

    private:
        std::vector<GpuTypedBufferPtr<GpuDrawElementsIndirectCommand>> drawCommands_;
        GpuTypedBufferPtr<GpuDrawElementsIndirectCommand> visibleCommands_;
        GpuTypedBufferPtr<GpuDrawElementsIndirectCommand> selectedLodCommands_;
        GpuTypedBufferPtr<glm::mat4> prevFrameModelTransforms_;
        GpuTypedBufferPtr<glm::mat4> modelTransforms_;
        GpuTypedBufferPtr<GpuAABB> aabbs_;
        GpuTypedBufferPtr<uint32_t> materialIndices_;
        std::unordered_map<RenderComponent *, std::unordered_map<MeshPtr, uint32_t>> drawCommandIndices_;
        std::unordered_map<RenderComponent *, std::unordered_set<MeshPtr>> pendingMeshUpdates_;

        RenderFaceCulling culling_;
        bool performedUpdate_ = false;
    };

    // This is used for things like GPU command generation where it takes a full CommandBuffer and
    // re-records it into a CommandReceiveBuffer
    struct GpuCommandReceiveBuffer {
        // Looks at the given command buffer and makes sure we have enough space to receive
        // elements from it
        void EnsureCapacity(const GpuCommandBufferPtr& buffer);

        GpuBuffer GetCommandBuffer() const;

        static inline GpuCommandReceiveBufferPtr Create() {
            return GpuCommandReceiveBufferPtr(new GpuCommandReceiveBuffer());
        }

    private:
        GpuBuffer receivedCommands_;
        size_t capacityBytes_ = 0;
    };

    // Manages a set of command buffers
    struct GpuCommandManager {
        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr> flatMeshes;
        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr> dynamicPbrMeshes;
        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr> staticPbrMeshes;

        GpuCommandManager(size_t numLods);

        size_t NumLods() const;

        void RecordCommands(const EntityPtr&, const GpuMaterialBufferPtr&);
        void RemoveAllCommands(const EntityPtr&);
        void ClearCommands();

        void UpdateTransforms(const EntityPtr&);
        void UpdateMaterials(const EntityPtr&, const GpuMaterialBufferPtr&);

        bool UploadFlatDataToGpu();
        bool UploadDynamicDataToGpu();
        bool UploadStaticDataToGpu();
        bool UploadDataToGpu();

        static inline GpuCommandManagerPtr Create(const size_t numLods) {
            return GpuCommandManagerPtr(new GpuCommandManager(numLods));
        }

    private:
        std::unordered_set<EntityPtr> entities_;
        size_t numLods_;
    };

    // Manages a set of command receivers
    struct GpuCommandReceiveManager {
        std::unordered_map<RenderFaceCulling, GpuCommandReceiveBufferPtr> flatMeshes;
        std::unordered_map<RenderFaceCulling, GpuCommandReceiveBufferPtr> dynamicPbrMeshes;
        std::unordered_map<RenderFaceCulling, GpuCommandReceiveBufferPtr> staticPbrMeshes;

        GpuCommandReceiveManager();

        void EnsureCapacity(const GpuCommandManagerPtr& manager);

        static inline GpuCommandReceiveManagerPtr Create() {
            return GpuCommandReceiveManagerPtr(new GpuCommandReceiveManager());
        }
    };
}