#include "StratusGpuCommandBuffer.h"

namespace stratus {
    GpuCommandBuffer::GpuCommandBuffer(const RenderFaceCulling& culling, usize numLods, usize commandBlockSize)
    {
        culling_ = culling;
        numLods = std::max<usize>(1, numLods);

        drawCommands_.resize(numLods);
        for (usize i = 0; i < numLods; ++i) {
            drawCommands_[i] = (GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true));
        }

        visibleCommands_ = GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true);
        selectedLodCommands_ = GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true);
        prevFrameModelTransforms_ = GpuTypedBuffer<glm::mat4>::Create(commandBlockSize, true);
        modelTransforms_ = GpuTypedBuffer<glm::mat4>::Create(commandBlockSize, true);
        aabbs_ = GpuTypedBuffer<GpuAABB>::Create(commandBlockSize, true);
        materialIndices_ = GpuTypedBuffer<u32>::Create(commandBlockSize, true);
    }

    usize GpuCommandBuffer::NumDrawCommands() const
    {
        return drawCommands_[0]->Size();
    }

    usize GpuCommandBuffer::NumLods() const
    {
        return drawCommands_.size();
    }

    usize GpuCommandBuffer::CommandCapacity() const {
        return drawCommands_[0]->Capacity();
    }

    const RenderFaceCulling& GpuCommandBuffer::GetFaceCulling() const
    {
        return culling_;
    }

    void GpuCommandBuffer::RecordCommand(
        RenderComponent* component,
        MeshWorldTransforms* transforms,
        const usize meshIndex,
        const usize materialIndex)
    {
        const MeshPtr mesh = component->GetMesh(meshIndex);

        // Face culling does not match this command buffer so can't record
        if (mesh->GetFaceCulling() != GetFaceCulling()) return;

        auto it = drawCommandIndices_.find(component);
        if (it == drawCommandIndices_.end()) {
            it = drawCommandIndices_.insert(std::make_pair(
                component,
                std::unordered_map<MeshletPtr, u32>()
            )).first;
        }

        for (usize i = 0; i < mesh->NumMeshlets(); ++i) {
            auto meshlet = mesh->GetMeshlet(i);

            // Command already exists for render component/mesh pair
            if (it->second.find(meshlet) != it->second.end()) {
                continue;
            }

            // Record the metadata 
            auto index = prevFrameModelTransforms_->Add(transforms->transforms[meshIndex]);
            modelTransforms_->Add(transforms->transforms[meshIndex]);
            auto aabb = meshlet->IsFinalized() ? meshlet->GetAABB() : GpuAABB();
            aabbs_->Add(aabb);
            materialIndices_->Add(materialIndex);

            // Record the lod commands
            visibleCommands_->Add(GpuDrawElementsIndirectCommand());
            selectedLodCommands_->Add(GpuDrawElementsIndirectCommand());

            for (usize lod = 0; lod < NumLods(); ++lod) {
                GpuDrawElementsIndirectCommand command;

                if (meshlet->IsFinalized()) {
                    command.baseInstance = 0;
                    command.baseVertex = 0;
                    command.firstIndex = meshlet->GetIndexOffset(lod);
                    command.instanceCount = 1;
                    command.vertexCount = meshlet->GetNumIndices(lod);
                }

                drawCommands_[lod]->Add(command);
            }

            it->second.insert(std::make_pair(meshlet, index));

            InsertMeshPending_(component, meshlet);

            performedUpdate_ = true;
        }
    }

    void GpuCommandBuffer::RemoveAllCommands(RenderComponent* component)
    {
        auto it = drawCommandIndices_.find(component);
        if (it == drawCommandIndices_.end()) {
            return;
        }

        pendingMeshUpdates_.erase(component);

        for (auto& entry : it->second) {
            const auto index = entry.second;

            // Remove top level data
            visibleCommands_->Remove(index);
            selectedLodCommands_->Remove(index);
            prevFrameModelTransforms_->Remove(index);
            modelTransforms_->Remove(index);
            aabbs_->Remove(index);
            materialIndices_->Remove(index);

            // Remove all lods
            for (usize i = 0; i < NumLods(); ++i) {
                drawCommands_[i]->Remove(index);
            }

            performedUpdate_ = true;
        }

        drawCommandIndices_.erase(component);
    }

    void GpuCommandBuffer::UpdateTransforms(RenderComponent* component, MeshWorldTransforms* transforms)
    {
        auto it = drawCommandIndices_.find(component);
        if (it == drawCommandIndices_.end()) {
            return;
        }

        for (usize i = 0; i < component->GetMeshCount(); ++i) {
            const MeshPtr mesh = component->GetMesh(i);

            for (usize j = 0; j < mesh->NumMeshlets(); ++j) {
                auto meshlet = mesh->GetMeshlet(j);
                auto index = it->second.find(meshlet);
                // Mesh does not match this command buffer
                if (index == it->second.end()) {
                    continue;
                }

                modelTransforms_->Set(transforms->transforms[i], index->second);
                performedUpdate_ = true;
            }
        }
    }

    void GpuCommandBuffer::UpdateMaterials(RenderComponent* component, const GpuMaterialBufferPtr& materials)
    {
        auto it = drawCommandIndices_.find(component);
        if (it == drawCommandIndices_.end()) {
            return;
        }

        for (usize i = 0; i < component->GetMeshCount(); ++i) {
            const MeshPtr mesh = component->GetMesh(i);

            for (usize j = 0; j < mesh->NumMeshlets(); ++j) {
                auto meshlet = mesh->GetMeshlet(j);
                auto index = it->second.find(meshlet);
                // Mesh does not match this command buffer
                if (index == it->second.end()) {
                    continue;
                }

                auto material = component->GetMaterialAt(i);
                materialIndices_->Set(materials->GetMaterialIndex(material), index->second);
                performedUpdate_ = true;
            }
        }
    }

    bool GpuCommandBuffer::UploadDataToGpu()
    {
        // Process pending meshes
        auto pending = std::move(pendingMeshUpdates_);
        for (auto& [component, meshes] : pending) {
            const auto& indices = drawCommandIndices_.find(component)->second;
            for (auto& mesh : meshes) {
                // Mesh is still not done
                if (InsertMeshPending_(component, mesh)) {
                    continue;
                }

                const auto index = indices.find(mesh)->second;
                performedUpdate_ = true;
                aabbs_->Set(mesh->GetAABB(), index);

                for (usize lod = 0; lod < NumLods(); ++lod) {
                    GpuDrawElementsIndirectCommand command;
                    command.baseInstance = 0;
                    command.baseVertex = 0;
                    command.firstIndex = mesh->GetIndexOffset(lod);
                    command.instanceCount = 1;
                    command.vertexCount = mesh->GetNumIndices(lod);

                    drawCommands_[lod]->Set(command, index);
                }
            }
        }

        // Don't need to upload for visibleCommands_ or selectedLodCommands_ since
        // they are meant to be directly modified on the GPU
        for (usize i = 0; i < NumLods(); ++i) {
            drawCommands_[i]->UploadChangesToGpu();
        }

        prevFrameModelTransforms_->UploadChangesToGpu();
        modelTransforms_->UploadChangesToGpu();
        aabbs_->UploadChangesToGpu();
        materialIndices_->UploadChangesToGpu();

        auto updated = performedUpdate_;
        performedUpdate_ = false;
        return updated;
    }

    void GpuCommandBuffer::BindMaterialIndicesBuffer(u32 index) const
    {
        auto buffer = materialIndices_->GetBuffer();
        if (buffer == GpuBuffer()) {
            throw std::runtime_error("Null material indices GpuBuffer");
        }
        buffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer::BindPrevFrameModelTransformBuffer(u32 index) const
    {
        auto buffer = prevFrameModelTransforms_->GetBuffer();
        if (buffer == GpuBuffer()) {
            throw std::runtime_error("Null previous frame model transform GpuBuffer");
        }
        buffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer::BindModelTransformBuffer(u32 index) const
    {
        auto buffer = modelTransforms_->GetBuffer();
        if (buffer == GpuBuffer()) {
            throw std::runtime_error("Null model transform GpuBuffer");
        }
        buffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer::BindAabbBuffer(u32 index) const
    {
        auto buffer = aabbs_->GetBuffer();
        if (buffer == GpuBuffer()) {
            throw std::runtime_error("Null aabb GpuBuffer");
        }
        buffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer::BindIndirectDrawCommands(const usize lod) const
    {
        if (lod >= NumLods() || drawCommands_[lod]->GetBuffer() == GpuBuffer()) {
            throw std::runtime_error("Null indirect draw command buffer");
        }
        drawCommands_[lod]->GetBuffer().Bind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);
    }

    void GpuCommandBuffer::UnbindIndirectDrawCommands(const usize lod) const
    {
        if (lod >= NumLods() || drawCommands_[lod]->GetBuffer() == GpuBuffer()) {
            throw std::runtime_error("Null indirect draw command buffer");
        }
        drawCommands_[lod]->GetBuffer().Unbind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);
    }

    GpuBuffer GpuCommandBuffer::GetIndirectDrawCommandsBuffer(const usize lod) const
    {
        if (lod >= NumLods()) {
            throw std::runtime_error("LOD requested exceeds max available LOD");
        }
        return drawCommands_[lod]->GetBuffer();
    }

    GpuBuffer GpuCommandBuffer::GetVisibleDrawCommandsBuffer() const
    {
        return visibleCommands_->GetBuffer();
    }

    GpuBuffer GpuCommandBuffer::GetSelectedLodDrawCommandsBuffer() const
    {
        return selectedLodCommands_->GetBuffer();
    }

    bool GpuCommandBuffer::InsertMeshPending_(RenderComponent* component, MeshletPtr meshlet)
    {
        if (!meshlet->IsFinalized()) {
            auto pending = pendingMeshUpdates_.find(component);
            if (pending == pendingMeshUpdates_.end()) {
                pending = pendingMeshUpdates_.insert(std::make_pair(
                    component,
                    std::unordered_set<MeshletPtr>()
                )).first;
            }

            pending->second.insert(meshlet);

            return true;
        }

        return false;
    }

    void GpuCommandReceiveBuffer::EnsureCapacity(const GpuCommandBufferPtr& buffer, usize copies) {
        if (copies == 0) return;

        const usize capacity = capacityBytes_ / sizeof(GpuDrawElementsIndirectCommand);
        const bool resize = capacity < (copies * buffer->CommandCapacity());

        if (resize) {
            capacityBytes_ = copies * buffer->CommandCapacity() * sizeof(GpuDrawElementsIndirectCommand);
            const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
            receivedCommands_ = GpuBuffer(nullptr, capacityBytes_, flags);
        }
    }

    GpuBuffer GpuCommandReceiveBuffer::GetCommandBuffer() const {
        return receivedCommands_;
    }

    GpuCommandManager::GpuCommandManager(usize numLods)
    {
        numLods = std::max<usize>(1, numLods);
        numLods_ = numLods;

        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        for (usize i = 0; i < 3; ++i) {
            auto fm = GpuCommandBuffer::Create(cullingValues[i], numLods, 10000);
            auto dm = GpuCommandBuffer::Create(cullingValues[i], numLods, 10000);
            auto sm = GpuCommandBuffer::Create(cullingValues[i], numLods, 10000);

            flatMeshes.insert(std::make_pair(cullingValues[i], fm));
            dynamicPbrMeshes.insert(std::make_pair(cullingValues[i], dm));
            staticPbrMeshes.insert(std::make_pair(cullingValues[i], sm));
        }
    }

    usize GpuCommandManager::NumLods() const
    {
        return numLods_;
    }

    static inline bool IsRenderable_(const EntityPtr& e) {
        return e->Components().ContainsComponent<RenderComponent>();
    }

    static inline bool IsLightInteracting_(const EntityPtr& e) {
        auto component = e->Components().GetComponent<LightInteractionComponent>();
        return component.status == EntityComponentStatus::COMPONENT_ENABLED;
    }

    static inline bool IsStaticEntity_(const EntityPtr& e) {
        auto sc = e->Components().GetComponent<StaticObjectComponent>();
        return sc.component != nullptr && sc.status == EntityComponentStatus::COMPONENT_ENABLED;
    }

    void GpuCommandManager::RecordCommands(const EntityPtr& e, const GpuMaterialBufferPtr& materials)
    {
        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        // Cannot be displayed
        if (!IsRenderable_(e)) {
            return;
        }

        // Already recorded
        if (entities_.find(e) != entities_.end()) {
            return;
        }

        entities_.insert(e);

        auto c = GetComponent<RenderComponent>(e);
        auto mt = GetComponent<MeshWorldTransforms>(e);

        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>* buffer;

        if (!IsLightInteracting_(e)) {
            buffer = &flatMeshes;
        }
        else {
            if (IsStaticEntity_(e)) {
                buffer = &staticPbrMeshes;
            }
            else {
                buffer = &dynamicPbrMeshes;
            }
        }

        for (usize i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            for (usize meshIndex = 0; meshIndex < c->GetMeshCount(); ++meshIndex) {
                auto materialIndex = materials->GetMaterialIndex(c->GetMaterialAt(meshIndex));
                buffer->find(cull)->second->RecordCommand(c, mt, meshIndex, materialIndex);
            }
        }
    }

    void GpuCommandManager::RemoveAllCommands(const EntityPtr& e)
    {
        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>* buffers[] = {
            &flatMeshes,
            &dynamicPbrMeshes,
            &staticPbrMeshes
        };

        if (entities_.find(e) == entities_.end()) {
            return;
        }

        entities_.erase(e);

        auto c = GetComponent<RenderComponent>(e);

        for (usize i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            for (usize k = 0; k < 3; ++k) {
                auto buffer = buffers[k];

                buffer->find(cull)->second->RemoveAllCommands(c);
            }
        }
    }

    void GpuCommandManager::ClearCommands()
    {
        for (auto& e : entities_) {
            RemoveAllCommands(e);
        }
    }

    void GpuCommandManager::UpdateTransforms(const EntityPtr& e)
    {
        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>* buffers[] = {
            &flatMeshes,
            &dynamicPbrMeshes,
            &staticPbrMeshes
        };

        if (entities_.find(e) == entities_.end()) {
            return;
        }

        auto c = GetComponent<RenderComponent>(e);
        auto mt = GetComponent<MeshWorldTransforms>(e);

        for (usize i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            for (usize k = 0; k < 3; ++k) {
                auto buffer = buffers[k];

                buffer->find(cull)->second->UpdateTransforms(c, mt);
            }
        }
    }

    void GpuCommandManager::UpdateMaterials(const EntityPtr& e, const GpuMaterialBufferPtr& materials)
    {
        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        std::unordered_map<RenderFaceCulling, GpuCommandBufferPtr>* buffers[] = {
            &flatMeshes,
            &dynamicPbrMeshes,
            &staticPbrMeshes
        };

        if (entities_.find(e) == entities_.end()) {
            return;
        }

        auto c = GetComponent<RenderComponent>(e);

        for (usize i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            for (usize k = 0; k < 3; ++k) {
                auto buffer = buffers[k];

                buffer->find(cull)->second->UpdateMaterials(c, materials);
            }
        }
    }

    bool GpuCommandManager::UploadFlatDataToGpu()
    {
        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        bool changed = false;

        for (usize i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            changed = changed || flatMeshes.find(cull)->second->UploadDataToGpu();
        }

        return changed;
    }

    bool GpuCommandManager::UploadDynamicDataToGpu()
    {
        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        bool changed = false;

        for (usize i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            changed = changed || dynamicPbrMeshes.find(cull)->second->UploadDataToGpu();
        }

        return changed;
    }

    bool GpuCommandManager::UploadStaticDataToGpu()
    {
        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        bool changed = false;

        for (usize i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            changed = changed || staticPbrMeshes.find(cull)->second->UploadDataToGpu();
        }

        return changed;
    }

    bool GpuCommandManager::UploadDataToGpu()
    {
        return UploadFlatDataToGpu() || UploadDynamicDataToGpu() || UploadStaticDataToGpu();
    }

    GpuCommandReceiveManager::GpuCommandReceiveManager() {
        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        for (usize i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            flatMeshes.insert(std::make_pair(cull, GpuCommandReceiveBuffer::Create()));
            dynamicPbrMeshes.insert(std::make_pair(cull, GpuCommandReceiveBuffer::Create()));
            staticPbrMeshes.insert(std::make_pair(cull, GpuCommandReceiveBuffer::Create()));
        }
    }

    void GpuCommandReceiveManager::EnsureCapacity(const GpuCommandManagerPtr& manager, usize copies) {
#define ENSURE_CAPACITY(readonly, writeonly)                          \
        for (const auto& [cull, buffer] : readonly) {                     \
            writeonly.find(cull)->second->EnsureCapacity(buffer, copies); \
        }

        ENSURE_CAPACITY(manager->flatMeshes, flatMeshes);
        ENSURE_CAPACITY(manager->dynamicPbrMeshes, dynamicPbrMeshes);
        ENSURE_CAPACITY(manager->staticPbrMeshes, staticPbrMeshes);

#undef ENSURE_CAPACITY
    }
}