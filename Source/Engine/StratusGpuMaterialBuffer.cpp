#include "StratusGpuMaterialBuffer.h"
#include "StratusResourceManager.h"
#include "StratusLog.h"

namespace stratus {
    GpuMaterialBuffer::GpuMaterialBuffer(size_t maxMaterials)
    {
        maxMaterials = std::max<size_t>(1, maxMaterials);
        const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
        materials_ = std::vector<GpuMaterial>(maxMaterials, GpuMaterial());
        gpuMaterials_ = GpuBuffer((const void *)materials_.data(), sizeof(GpuMaterial) * maxMaterials, flags);

        for (uint32_t i = 0; i < uint32_t(maxMaterials); ++i) {
            freeIndices_.push_back(i);
        }
    }

    GpuMaterialBuffer::~GpuMaterialBuffer()
    {
    }

    static inline bool ValidateTexture(const Texture& tex, const TextureLoadingStatus& status) {
        return status == TextureLoadingStatus::LOADING_DONE;
    }

    void GpuMaterialBuffer::CopyMaterialToGpuStaging_(const MaterialPtr& material, const int index) {

        UpdateModifiedIndices_(index);
        GpuMaterial * gpuMaterial = &materials_[index];

        gpuMaterial->flags = 0;

        SET_FLOAT4(gpuMaterial->diffuseColor, material->GetDiffuseColor());
        SET_FLOAT3(gpuMaterial->emissiveColor, material->GetEmissiveColor());
        SET_FLOAT3(gpuMaterial->baseReflectivity, material->GetBaseReflectivity());
        SET_FLOAT3(gpuMaterial->maxReflectivity, material->GetMaxReflectivity());
        SET_FLOAT2(gpuMaterial->metallicRoughness, glm::vec2(material->GetMetallic(), material->GetRoughness()));

        auto diffuseHandle = material->GetDiffuseTexture();
        auto ambientHandle = material->GetEmissiveTexture();
        auto normalHandle = material->GetNormalMap();
        auto depthHandle = material->GetDepthMap();
        auto roughnessHandle = material->GetRoughnessMap();
        auto metallicHandle = material->GetMetallicMap();
        auto metallicRoughnessHandle = material->GetMetallicRoughnessMap();

        TextureLoadingStatus diffuseStatus;
        auto diffuse = INSTANCE(ResourceManager)->LookupTexture(diffuseHandle, diffuseStatus);
        TextureLoadingStatus ambientStatus;
        auto ambient = INSTANCE(ResourceManager)->LookupTexture(ambientHandle, ambientStatus);
        TextureLoadingStatus normalStatus;
        auto normal = INSTANCE(ResourceManager)->LookupTexture(normalHandle, normalStatus);
        TextureLoadingStatus depthStatus;
        auto depth = INSTANCE(ResourceManager)->LookupTexture(depthHandle, depthStatus);
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

        if (ValidateTexture(ambient, ambientStatus)) {
            gpuMaterial->emissiveMap = ambient.GpuHandle();
            gpuMaterial->flags |= GPU_EMISSIVE_MAPPED;
            resident.push_back(TextureMemResidencyGuard(ambient));
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (ambientHandle != TextureHandle::Null() && ambientStatus != TextureLoadingStatus::FAILED) {
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

        if (ValidateTexture(depth, depthStatus)) {
            gpuMaterial->depthMap = depth.GpuHandle();
            gpuMaterial->flags |= GPU_DEPTH_MAPPED;
            resident.push_back(TextureMemResidencyGuard(depth));
        }
        // If this is true then the texture is still loading so we need to check again later
        else if (depthHandle != TextureHandle::Null() && depthStatus != TextureLoadingStatus::FAILED) {
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
    }

    void GpuMaterialBuffer::MarkMaterialsUsed(RenderComponent * component)
    {
        for (size_t i = 0; i < component->GetMaterialCount(); ++i) {
            auto material = component->GetMaterialAt(i);

            auto it = usedIndices_.find(material);
            uint32_t index;
            // No materials currently reference this material so add a new entry
            if (it == usedIndices_.end()) {
                if (freeIndices_.size() == 0) {
                    throw std::runtime_error("Ran out of open material slots");
                }

                index = freeIndices_.front();
                freeIndices_.pop_front();

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

                freeIndices_.push_back(index);
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
        if (firstModifiedMaterialIndex_ != lastModifiedMaterialIndex_) {
            const intptr_t offsetBytes = intptr_t(firstModifiedMaterialIndex_) * sizeof(GpuMaterial);
            const uintptr_t sizeBytes = uintptr_t(lastModifiedMaterialIndex_ - firstModifiedMaterialIndex_) * sizeof(GpuMaterial);
            const void * data = (const void *)(materials_.data() + firstModifiedMaterialIndex_);
            gpuMaterials_.CopyDataToBuffer(offsetBytes, sizeBytes, data);

            //const uintptr_t sizeBytes = (uintptr_t)(lastModifiedMaterialIndex_) * sizeof(GpuMaterial);
            //const uintptr_t sizeBytes = materials_.size() * sizeof(GpuMaterial);
            //gpuMaterials_.CopyDataToBuffer(0, sizeBytes, (const void *)materials_.data());

            firstModifiedMaterialIndex_ = -1;
            lastModifiedMaterialIndex_ = -1;
        }

        auto pending = std::move(pendingMaterials_);
        for (auto& p : pending) {
            const int index = static_cast<int>(usedIndices_.find(p)->second);
            CopyMaterialToGpuStaging_(p, index);
        }
    }

    GpuBuffer GpuMaterialBuffer::GetMaterialBuffer() const
    {
        return gpuMaterials_;
    }

    void GpuMaterialBuffer::UpdateModifiedIndices_(const int index)
    {
        if (index <= firstModifiedMaterialIndex_ || firstModifiedMaterialIndex_ == -1) {
            firstModifiedMaterialIndex_ = index;
        }

        if (lastModifiedMaterialIndex_ <= index) {
            lastModifiedMaterialIndex_ = index + 1;
        }
    }
}