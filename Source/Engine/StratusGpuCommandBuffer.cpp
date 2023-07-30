#include "StratusGpuCommandBuffer.h"

namespace stratus {
    GpuCommandBuffer2::GpuCommandBuffer2(const RenderFaceCulling& culling, size_t numLods, size_t commandBlockSize)
    {
        culling_ = culling;
        numLods = std::max<size_t>(1, numLods);

        drawCommands_.resize(numLods);
        for (size_t i = 0; i < numLods; ++i) {
            drawCommands_[i] = (GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true));
        }

        visibleCommands_ = GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true);
        selectedLodCommands_ = GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true);
        visibleLowestLodCommands_ = GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true);
        prevFrameModelTransforms_ = GpuTypedBuffer<glm::mat4>::Create(commandBlockSize, true);
        modelTransforms_ = GpuTypedBuffer<glm::mat4>::Create(commandBlockSize, true);
        aabbs_ = GpuTypedBuffer<GpuAABB>::Create(commandBlockSize, true);
        materialIndices_ = GpuTypedBuffer<uint32_t>::Create(commandBlockSize, true);
    }

    size_t GpuCommandBuffer2::NumDrawCommands() const
    {
        return drawCommands_[0]->Size();
    }

    size_t GpuCommandBuffer2::NumLods() const
    {
        return drawCommands_.size();
    }

    const RenderFaceCulling& GpuCommandBuffer2::GetFaceCulling() const
    {
        return culling_;
    }

    void GpuCommandBuffer2::RecordCommand(
        RenderComponent* component, 
        MeshWorldTransforms* transforms,
        const size_t meshIndex,
        const size_t materialIndex)
    {
        const MeshPtr mesh = component->GetMesh(meshIndex);

        // Face culling does not match this command buffer so can't record
        if (mesh->GetFaceCulling() != GetFaceCulling()) return;

        auto it = drawCommandIndices_.find(component);
        if (it == drawCommandIndices_.end()) {
            it = drawCommandIndices_.insert(std::make_pair(
                component,
                std::unordered_map<MeshPtr, uint32_t>()
            )).first;
        }
        // Command already exists for render component/mesh pair
        else if (it->second.find(mesh) != it->second.end()) {
            return;
        }

        // Record the metadata 
        auto index = prevFrameModelTransforms_->Add(transforms->transforms[meshIndex]);
        modelTransforms_->Add(transforms->transforms[meshIndex]);
        auto aabb = mesh->IsFinalized() ? mesh->GetAABB() : GpuAABB();
        aabbs_->Add(aabb);
        materialIndices_->Add(materialIndex);

        // Record the lod commands
        visibleCommands_->Add(GpuDrawElementsIndirectCommand());
        selectedLodCommands_->Add(GpuDrawElementsIndirectCommand());
        visibleLowestLodCommands_->Add(GpuDrawElementsIndirectCommand());

        for (size_t lod = 0; lod < NumLods(); ++lod) {
            GpuDrawElementsIndirectCommand command;

            if (mesh->IsFinalized()) {
                command.baseInstance = 0;
                command.baseVertex = 0;
                command.firstIndex = mesh->GetIndexOffset(lod);
                command.instanceCount = 1;
                command.vertexCount = mesh->GetNumIndices(lod);
            }

            drawCommands_[lod]->Add(command);
        }

        it->second.insert(std::make_pair(mesh, index));

        InsertMeshPending_(component, mesh);

        performedUpdate_ = true;
    }

    void GpuCommandBuffer2::RemoveAllCommands(RenderComponent* component)
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
            visibleLowestLodCommands_->Remove(index);
            prevFrameModelTransforms_->Remove(index);
            modelTransforms_->Remove(index);
            aabbs_->Remove(index);
            materialIndices_->Remove(index);

            // Remove all lods
            for (size_t i = 0; i < NumLods(); ++i) {
                drawCommands_[i]->Remove(index);
            }

            performedUpdate_ = true;
        }

        drawCommandIndices_.erase(component);
    }

    void GpuCommandBuffer2::UpdateTransforms(RenderComponent* component, MeshWorldTransforms* transforms)
    {
        auto it = drawCommandIndices_.find(component);
        if (it == drawCommandIndices_.end()) {
            return;
        }

        for (size_t i = 0; i < component->GetMeshCount(); ++i) {
            const MeshPtr mesh = component->GetMesh(i);
            auto index = it->second.find(mesh);
            // Mesh does not match this command buffer
            if (index == it->second.end()) {
                continue;
            }

            modelTransforms_->Set(transforms->transforms[i], index->second);
            performedUpdate_ = true;
        }
    }

    void GpuCommandBuffer2::UpdateMaterials(RenderComponent* component, const GpuMaterialBufferPtr& materials)
    {
        auto it = drawCommandIndices_.find(component);
        if (it == drawCommandIndices_.end()) {
            return;
        }

        for (size_t i = 0; i < component->GetMeshCount(); ++i) {
            const MeshPtr mesh = component->GetMesh(i);
            auto index = it->second.find(mesh);
            // Mesh does not match this command buffer
            if (index == it->second.end()) {
                continue;
            }

            auto material = component->GetMaterialAt(i);
            materialIndices_->Set(materials->GetMaterialIndex(material), index->second);
            performedUpdate_ = true;
        }
    }

    bool GpuCommandBuffer2::UploadDataToGpu()
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

                for (size_t lod = 0; lod < NumLods(); ++lod) {
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
        for (size_t i = 0; i < NumLods(); ++i) {
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

    void GpuCommandBuffer2::BindMaterialIndicesBuffer(uint32_t index)
    {
        auto buffer = materialIndices_->GetBuffer();
        if (buffer == GpuBuffer()) {
            throw std::runtime_error("Null material indices GpuBuffer");
        }
        buffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer2::BindPrevFrameModelTransformBuffer(uint32_t index)
    {
        auto buffer = prevFrameModelTransforms_->GetBuffer();
        if (buffer == GpuBuffer()) {
            throw std::runtime_error("Null previous frame model transform GpuBuffer");
        }
        buffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer2::BindModelTransformBuffer(uint32_t index)
    {
        auto buffer = modelTransforms_->GetBuffer();
        if (buffer == GpuBuffer()) {
            throw std::runtime_error("Null model transform GpuBuffer");
        }
        buffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer2::BindAabbBuffer(uint32_t index)
    {
        auto buffer = aabbs_->GetBuffer();
        if (buffer == GpuBuffer()) {
            throw std::runtime_error("Null aabb GpuBuffer");
        }
        buffer.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer2::BindIndirectDrawCommands(const size_t lod)
    {
        if (lod >= NumLods() || drawCommands_[lod]->GetBuffer() == GpuBuffer()) {
            throw std::runtime_error("Null indirect draw command buffer");
        }
        drawCommands_[lod]->GetBuffer().Bind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);
    }

    void GpuCommandBuffer2::UnbindIndirectDrawCommands(const size_t lod)
    {
        if (lod >= NumLods() || drawCommands_[lod]->GetBuffer() == GpuBuffer()) {
            throw std::runtime_error("Null indirect draw command buffer");
        }
        drawCommands_[lod]->GetBuffer().Unbind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);
    }

    GpuBuffer GpuCommandBuffer2::GetIndirectDrawCommandsBuffer(const size_t lod) const
    {
        if (lod >= NumLods()) {
            throw std::runtime_error("LOD requested exceeds max available LOD");
        }
        return drawCommands_[lod]->GetBuffer();
    }

    GpuBuffer GpuCommandBuffer2::GetVisibleDrawCommandsBuffer() const
    {
        return visibleCommands_->GetBuffer();
    }

    GpuBuffer GpuCommandBuffer2::GetSelectedLodDrawCommandsBuffer() const
    {
        return selectedLodCommands_->GetBuffer();
    }

    GpuBuffer GpuCommandBuffer2::GetVisibleLowestLodDrawCommandsBuffer() const {
        return visibleLowestLodCommands_->GetBuffer();
    }

    bool GpuCommandBuffer2::InsertMeshPending_(RenderComponent* component, MeshPtr mesh)
    {
        if (!mesh->IsFinalized()) {
            auto pending = pendingMeshUpdates_.find(component);
            if (pending == pendingMeshUpdates_.end()) {
                pending = pendingMeshUpdates_.insert(std::make_pair(
                    component,
                    std::unordered_set<MeshPtr>()
                )).first;
            }

            pending->second.insert(mesh);

            return true;
        }

        return false;
    }

    GpuCommandManager::GpuCommandManager(size_t numLods)
    {
        numLods = std::max<size_t>(1, numLods);
        numLods_ = numLods;

        static constexpr RenderFaceCulling cullingValues[] = {
            RenderFaceCulling::CULLING_CCW,
            RenderFaceCulling::CULLING_CW,
            RenderFaceCulling::CULLING_NONE
        };

        for (size_t i = 0; i < 3; ++i) {
            auto fm = GpuCommandBuffer2::Create(cullingValues[i], numLods, 10000);
            auto dm = GpuCommandBuffer2::Create(cullingValues[i], numLods, 10000);
            auto sm = GpuCommandBuffer2::Create(cullingValues[i], numLods, 10000);

            flatMeshes.insert(std::make_pair(cullingValues[i], fm));
            dynamicPbrMeshes.insert(std::make_pair(cullingValues[i], dm));
            staticPbrMeshes.insert(std::make_pair(cullingValues[i], sm));
        }
    }

    size_t GpuCommandManager::NumLods() const
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
        
        std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>* buffer;

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

        for (size_t i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            for (size_t meshIndex = 0; meshIndex < c->GetMeshCount(); ++meshIndex) {
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

        std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr> * buffers[] = {
            &flatMeshes,
            &dynamicPbrMeshes,
            &staticPbrMeshes
        };

        if (entities_.find(e) == entities_.end()) {
            return;
        }

        entities_.erase(e);

        auto c = GetComponent<RenderComponent>(e);

        for (size_t i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            for (size_t k = 0; k < 3; ++k) {
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

        std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>* buffers[] = {
            &flatMeshes,
            &dynamicPbrMeshes,
            &staticPbrMeshes
        };

        if (entities_.find(e) == entities_.end()) {
            return;
        }

        auto c = GetComponent<RenderComponent>(e);
        auto mt = GetComponent<MeshWorldTransforms>(e);

        for (size_t i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            for (size_t k = 0; k < 3; ++k) {
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

        std::unordered_map<RenderFaceCulling, GpuCommandBuffer2Ptr>* buffers[] = {
            &flatMeshes,
            &dynamicPbrMeshes,
            &staticPbrMeshes
        };

        if (entities_.find(e) == entities_.end()) {
            return;
        }

        auto c = GetComponent<RenderComponent>(e);

        for (size_t i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            for (size_t k = 0; k < 3; ++k) {
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

        for (size_t i = 0; i < 3; ++i) {
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

        for (size_t i = 0; i < 3; ++i) {
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

        for (size_t i = 0; i < 3; ++i) {
            const auto cull = cullingValues[i];
            changed = changed || staticPbrMeshes.find(cull)->second->UploadDataToGpu();
        }

        return changed;
    }

    bool GpuCommandManager::UploadDataToGpu()
    {
        return UploadFlatDataToGpu() || UploadDynamicDataToGpu() || UploadStaticDataToGpu();
    }
}