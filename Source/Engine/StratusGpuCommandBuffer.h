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
#include "StratusTypes.h"

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
        GpuCommandBuffer(const RenderFaceCulling&, usize numLods, usize commandBlockSize);

        GpuCommandBuffer(GpuCommandBuffer&&) = default;
        GpuCommandBuffer(const GpuCommandBuffer&) = delete;

        GpuCommandBuffer& operator=(GpuCommandBuffer&&) = delete;
        GpuCommandBuffer& operator=(const GpuCommandBuffer&) = delete;

        usize NumDrawCommands() const;
        usize NumLods() const;
        usize CommandCapacity() const;
        const RenderFaceCulling& GetFaceCulling() const;

        void RecordCommand(RenderComponent*, MeshWorldTransforms*, const usize, const usize);
        void RemoveAllCommands(RenderComponent*);

        void UpdateTransforms(RenderComponent*, MeshWorldTransforms*);
        void UpdateMaterials(RenderComponent*, const GpuMaterialBufferPtr&);

        bool UploadDataToGpu();

        void BindMaterialIndicesBuffer(u32 index) const;
        void BindPrevFrameModelTransformBuffer(u32 index) const;
        void BindModelTransformBuffer(u32 index) const;
        void BindAabbBuffer(u32 index) const;

        void BindIndirectDrawCommands(const usize lod) const;
        void UnbindIndirectDrawCommands(const usize lod) const;

        GpuBuffer GetIndirectDrawCommandsBuffer(const usize lod) const;
        GpuBuffer GetVisibleDrawCommandsBuffer() const;
        GpuBuffer GetSelectedLodDrawCommandsBuffer() const;

        static inline GpuCommandBufferPtr Create(const RenderFaceCulling& cull, const usize numLods, const usize commandBlockSize) {
            return GpuCommandBufferPtr(new GpuCommandBuffer(cull, numLods, commandBlockSize));
        }

    private:
        bool InsertMeshPending_(RenderComponent*, MeshletPtr);

    private:
        std::vector<GpuTypedBufferPtr<GpuDrawElementsIndirectCommand>> drawCommands_;
        GpuTypedBufferPtr<GpuDrawElementsIndirectCommand> visibleCommands_;
        GpuTypedBufferPtr<GpuDrawElementsIndirectCommand> selectedLodCommands_;
        GpuTypedBufferPtr<glm::mat4> prevFrameModelTransforms_;
        GpuTypedBufferPtr<glm::mat4> modelTransforms_;
        GpuTypedBufferPtr<GpuAABB> aabbs_;
        GpuTypedBufferPtr<u32> materialIndices_;
        std::unordered_map<RenderComponent *, std::unordered_map<MeshletPtr, u32>> drawCommandIndices_;
        std::unordered_map<RenderComponent *, std::unordered_set<MeshletPtr>> pendingMeshUpdates_;

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
        usize capacityBytes_ = 0;
    };

    // Manages a set of command buffers
    struct GpuCommandManager {
        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr> flatMeshes;
        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr> dynamicPbrMeshes;
        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr> staticPbrMeshes;

        GpuCommandManager(usize numLods);

        usize NumLods() const;

        void RecordCommands(const EntityPtr&, const GpuMaterialBufferPtr&);
        void RemoveAllCommands(const EntityPtr&);
        void ClearCommands();

        void UpdateTransforms(const EntityPtr&);
        void UpdateMaterials(const EntityPtr&, const GpuMaterialBufferPtr&);

        bool UploadFlatDataToGpu();
        bool UploadDynamicDataToGpu();
        bool UploadStaticDataToGpu();
        bool UploadDataToGpu();

        static inline GpuCommandManagerPtr Create(const usize numLods) {
            return GpuCommandManagerPtr(new GpuCommandManager(numLods));
        }

    private:
        std::unordered_set<EntityPtr> entities_;
        usize numLods_;
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