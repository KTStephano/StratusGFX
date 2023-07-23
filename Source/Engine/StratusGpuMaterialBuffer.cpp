#include "StratusGpuMaterialBuffer.h"
#include "StratusResourceManager.h"
#include "StratusLog.h"

namespace stratus {
    GpuMaterialBuffer::GpuMaterialBuffer(size_t maxMaterials)
    {
        maxMaterials = std::max<size_t>(1, maxMaterials);
        materials_ = GpuTypedBuffer<GpuMaterial>::Create(maxMaterials, false);
    }

    GpuMaterialBuffer::~GpuMaterialBuffer()
    {
    }

    static inline bool ValidateTexture(const Texture& tex, const TextureLoadingStatus& status) {
        return status == TextureLoadingStatus::LOADING_DONE;
    }

    void GpuMaterialBuffer::CopyMaterialToGpuStaging_(const MaterialPtr& material, const int index) {

        auto mat = materials_->GetRead(index);
        GpuMaterial* gpuMaterial = &mat;

        gpuMaterial->flags = 0;

        SET_FLOAT4(gpuMaterial->diffuseColor, material->GetDiffuseColor());
        SET_FLOAT3(gpuMaterial->emissiveColor, material->GetEmissiveColor());
        gpuMaterial->reflectance = material->GetReflectance();
        SET_FLOAT2(gpuMaterial->metallicRoughness, glm::vec2(material->GetMetallic(), material->GetRoughness()));

        auto diffuseHandle = material->GetDiffuseMap();
        auto emissiveHandle = material->GetEmissiveMap();
        auto normalHandle = material->GetNormalMap();
        auto roughnessHandle = material->GetRoughnessMap();
        auto metallicHandle = material->GetMetallicMap();
        auto metallicRoughnessHandle = material->GetMetallicRoughnessMap();

        TextureLoadingStatus diffuseStatus;
        auto diffuse = INSTANCE(ResourceManager)->LookupTexture(diffuseHandle, diffuseStatus);
        TextureLoadingStatus emissiveStatus;
        auto emissive = INSTANCE(ResourceManager)->LookupTexture(emissiveHandle, emissiveStatus);
        TextureLoadingStatus normalStatus;
        auto normal = INSTANCE(ResourceManager)->LookupTexture(normalHandle, normalStatus);
        TextureLoadingStatus roughnessStatus;
        auto roughness = INSTANCE(ResourceManager)->LookupTexture(roughnessHandle, roughnessStatus);
        TextureLoadingStatus metallicStatus;
        auto metallic = INSTANCE(ResourceManager)->LookupTexture(metallicHandle, metallicStatus);
        TextureLoadingStatus metallicRoughnessStatus;
        auto metallicRoughness = INSTANCE(ResourceManager)->LookupTexture(metallicRoughnessHandle, metallicRoughnessStatus);

        auto& resident = residentTexturesPerMaterial_.find(material)->second;

        if (ValidateTexture(diffuse, diffuseStatus)) {
            gpuMaterial->diffuseMap = diffuse.GpuHandle();
            gpuMaterial->flags |= GPU_DIFFUSE_MAPPED;
            resident.push_back(TextureMemResidencyGuard(diffuse));
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (diffuseHandle != TextureHandle::Null() && diffuseStatus != TextureLoadingStatus::FAILED) {
            pendingMaterials_.insert(material);
        }

        if (ValidateTexture(emissive, emissiveStatus)) {
            gpuMaterial->emissiveMap = emissive.GpuHandle();
            gpuMaterial->flags |= GPU_EMISSIVE_MAPPED;
            resident.push_back(TextureMemResidencyGuard(emissive));
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (emissiveHandle != TextureHandle::Null() && emissiveStatus != TextureLoadingStatus::FAILED) {
            pendingMaterials_.insert(material);
        }

        if (ValidateTexture(normal, normalStatus)) {
            gpuMaterial->normalMap = normal.GpuHandle();
            gpuMaterial->flags |= GPU_NORMAL_MAPPED;
            resident.push_back(TextureMemResidencyGuard(normal));
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (normalHandle != TextureHandle::Null() && normalStatus != TextureLoadingStatus::FAILED) {
            pendingMaterials_.insert(material);
        }

        if (ValidateTexture(roughness, roughnessStatus)) {
            gpuMaterial->roughnessMap = roughness.GpuHandle();
            gpuMaterial->flags |= GPU_ROUGHNESS_MAPPED;
            resident.push_back(TextureMemResidencyGuard(roughness));
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (roughnessHandle != TextureHandle::Null() && roughnessStatus != TextureLoadingStatus::FAILED) {
            pendingMaterials_.insert(material);
        }

        if (ValidateTexture(metallic, metallicStatus)) {
            gpuMaterial->metallicMap = metallic.GpuHandle();
            gpuMaterial->flags |= GPU_METALLIC_MAPPED;
            resident.push_back(TextureMemResidencyGuard(metallic));
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (metallicHandle != TextureHandle::Null() && metallicStatus != TextureLoadingStatus::FAILED) {
            pendingMaterials_.insert(material);
        }

        if (ValidateTexture(metallicRoughness, metallicRoughnessStatus)) {
            gpuMaterial->metallicRoughnessMap = metallicRoughness.GpuHandle();
            gpuMaterial->flags |= GPU_METALLIC_ROUGHNESS_MAPPED;
            resident.push_back(TextureMemResidencyGuard(metallicRoughness));
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (metallicRoughnessHandle != TextureHandle::Null() && metallicRoughnessStatus != TextureLoadingStatus::FAILED) {
            pendingMaterials_.insert(material);
        }

        materials_->Set(*gpuMaterial, index);
    }

    void GpuMaterialBuffer::MarkMaterialsUsed(RenderComponent * component)
    {
        for (size_t i = 0; i < component->GetMaterialCount(); ++i) {
            auto material = component->GetMaterialAt(i);

            auto it = usedIndices_.find(material);
            uint32_t index;
            // No materials currently reference this material so add a new entry
            if (it == usedIndices_.end()) {
                index = materials_->Add(GpuMaterial());

                usedIndices_.insert(std::make_pair(material, index));
                availableMaterials_.insert(std::make_pair(material, std::unordered_set<RenderComponent *>()));
                residentTexturesPerMaterial_.insert(std::make_pair(material, std::vector<TextureMemResidencyGuard>()));

                CopyMaterialToGpuStaging_(material, static_cast<int>(index));
            }
            else {
                index = it->second;
            }

            availableMaterials_.find(material)->second.insert(component);
        }
    }

    void GpuMaterialBuffer::MarkMaterialsUnused(RenderComponent * component)
    {
        for (size_t i = 0; i < component->GetMaterialCount(); ++i) {
            auto material = component->GetMaterialAt(i);
            auto mcit = availableMaterials_.find(material);
            if (mcit == availableMaterials_.end()) continue;

            mcit->second.erase(component);
            // No components reference this material anymore so remove it
            if (mcit->second.size() == 0) {
                const auto index = usedIndices_.find(material)->second;

                materials_->Remove(index);
                usedIndices_.erase(material);

                availableMaterials_.erase(material);
                residentTexturesPerMaterial_.erase(material);
                pendingMaterials_.erase(material);
            }
        }
    }

    uint32_t GpuMaterialBuffer::GetMaterialIndex(const MaterialPtr material) const
    {
        auto it = usedIndices_.find(material);
        if (it == usedIndices_.end()) {
            throw std::runtime_error("Material not found");
        }

        return it->second;
    }

    void GpuMaterialBuffer::UploadDataToGpu()
    {
        auto pending = std::move(pendingMaterials_);
        for (auto& p : pending) {
            const int index = static_cast<int>(usedIndices_.find(p)->second);
            CopyMaterialToGpuStaging_(p, index);
        }

        materials_->UploadChangesToGpu();
    }

    GpuBuffer GpuMaterialBuffer::GetMaterialBuffer() const
    {
        return materials_->GetBuffer();
    }
}