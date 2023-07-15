#include "StratusGpuCommandBuffer.h"

namespace stratus {
    GpuCommandBuffer2::GpuCommandBuffer2(const RenderFaceCulling& culling, size_t numLods, size_t commandBlockSize)
    {
        culling_ = culling;
        numLods = std::max<size_t>(1, numLods);

        drawCommands_.resize(numLods);
        for (size_t i = 0; numLods; ++i) {
            drawCommands_[i] = GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true);
        }

        visibleCommands_ = GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true);
        selectedLodCommands_ = GpuTypedBuffer<GpuDrawElementsIndirectCommand>::Create(commandBlockSize, true);
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
        const RenderComponent* component, 
        const MeshWorldTransforms* transforms,
        const size_t meshIndex,
        const size_t materialIndex)
    {
        const MeshPtr mesh = component->GetMesh(meshIndex);

        if (mesh->GetFaceCulling() != GetFaceCulling()) return;

        auto it = drawCommandIndices_.find(component);
        if (it == drawCommandIndices_.end()) {
            it = drawCommandIndices_.insert(std::make_pair(
                component,
                std::unordered_map<const MeshPtr, uint32_t>()
            )).first;
        }
        // Command already exists for render component/mesh pair
        else if (it->second.find(mesh) != it->second.end()) {
            return;
        }

        // Record the metadata 
        auto index = prevFrameModelTransforms_->Add(transforms->transforms[meshIndex]);
        modelTransforms_->Add(transforms->transforms[meshIndex]);
        aabbs_->Add(mesh->GetAABB());
        materialIndices_->Add(materialIndex);

        // Record the lod commands
        visibleCommands_->Add(GpuDrawElementsIndirectCommand());
        selectedLodCommands_->Add(GpuDrawElementsIndirectCommand());

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
    }

    void GpuCommandBuffer2::RemoveAllCommands(const RenderComponent* component)
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
            for (size_t i = 0; i < NumLods(); ++i) {
                drawCommands_[i]->Remove(index);
            }
        }

        drawCommandIndices_.erase(component);
    }

    void GpuCommandBuffer2::UpdateTransforms(const RenderComponent* component, const MeshWorldTransforms* transforms)
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
        }
    }

    void GpuCommandBuffer2::UpdateMaterials(const RenderComponent* component, const GpuMaterialBufferPtr& materials)
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
        }
    }

    void GpuCommandBuffer2::UploadDataToGpu()
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
    }

    void GpuCommandBuffer2::BindMaterialIndicesBuffer(uint32_t index)
    {
    }
    void GpuCommandBuffer2::BindPrevFrameModelTransformBuffer(uint32_t index)
    {
    }
    void GpuCommandBuffer2::BindModelTransformBuffer(uint32_t index)
    {
    }
    void GpuCommandBuffer2::BindAabbBuffer(uint32_t index)
    {
    }
    void GpuCommandBuffer2::BindIndirectDrawCommands(const size_t lod)
    {
    }
    void GpuCommandBuffer2::UnbindIndirectDrawCommands(const size_t lod)
    {
    }
    const GpuBuffer& GpuCommandBuffer2::GetIndirectDrawCommandsBuffer(const size_t lod) const
    {
        // TODO: insert return statement here
    }
    const GpuBuffer& GpuCommandBuffer2::GetVisibleDrawCommandsBuffer() const
    {
        // TODO: insert return statement here
    }
    const GpuBuffer& GpuCommandBuffer2::GetSelectedLodDrawCommandsBuffer() const
    {
        // TODO: insert return statement here
    }

    bool GpuCommandBuffer2::InsertMeshPending_(const RenderComponent* component, const MeshPtr mesh)
    {
        if (!mesh->IsFinalized()) {
            auto pending = pendingMeshUpdates_.find(component);
            if (pending == pendingMeshUpdates_.end()) {
                pending = pendingMeshUpdates_.insert(std::make_pair(
                    component,
                    std::unordered_set<const MeshPtr>()
                )).first;
            }

            pending->second.insert(mesh);

            return true;
        }

        return false;
    }
}