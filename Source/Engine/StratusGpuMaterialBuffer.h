#pragma once

#include "StratusGpuCommon.h"
#include "StratusMaterial.h"
#include "StratusGpuBuffer.h"
#include "StratusTexture.h"
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include "StratusRenderComponents.h"
#include "StratusEntity.h"

namespace stratus {
    struct GpuMaterialBuffer;
    typedef std::shared_ptr<GpuMaterialBuffer> GpuMaterialBufferPtr;

    // This class manages the current active materials in GPU memory
    struct GpuMaterialBuffer {
        GpuMaterialBuffer(size_t maxMaterials);
        
        GpuMaterialBuffer(GpuMaterialBuffer&&) = default;
        GpuMaterialBuffer(const GpuMaterialBuffer&) = delete;

        GpuMaterialBuffer& operator=(GpuMaterialBuffer&&) = delete;
        GpuMaterialBuffer& operator=(const GpuMaterialBuffer&) = delete;

        ~GpuMaterialBuffer();

        void MarkMaterialsUsed(RenderComponent *);
        void MarkMaterialsUnused(RenderComponent *);

        uint32_t GetMaterialIndex(const MaterialPtr) const;
        GpuBuffer GetMaterialBuffer() const;

        void UploadDataToGpu();

        static GpuMaterialBufferPtr Create(const size_t maxMaterials) {
            return GpuMaterialBufferPtr(new GpuMaterialBuffer(maxMaterials));
        }

    private:
        void UpdateModifiedIndices_(const int index);
        void CopyMaterialToGpuStaging_(const MaterialPtr& material, const int index);

    private:
        GpuBuffer gpuMaterials_;
        std::vector<GpuMaterial> materials_;
        // These are the materials we draw from to calculate the material-indices map
        std::unordered_map<MaterialPtr, std::unordered_set<RenderComponent *>> availableMaterials_;
        std::unordered_map<MaterialPtr, std::vector<TextureMemResidencyGuard>> residentTexturesPerMaterial_;
        // Indices can change completely if new materials are added
        std::unordered_map<MaterialPtr, uint32_t> usedIndices_;
        std::unordered_set<MaterialPtr> pendingMaterials_;
        std::list<uint32_t> freeIndices_;

        int firstModifiedMaterialIndex_ = -1;
        int lastModifiedMaterialIndex_ = -1;
    };
}