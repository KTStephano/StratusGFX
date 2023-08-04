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
#include "StratusTypes.h"

namespace stratus {
    struct GpuMaterialBuffer;
    typedef std::shared_ptr<GpuMaterialBuffer> GpuMaterialBufferPtr;

    // This class manages the current active materials in GPU memory
    struct GpuMaterialBuffer {
        GpuMaterialBuffer(usize maxMaterials);
        
        GpuMaterialBuffer(GpuMaterialBuffer&&) = default;
        GpuMaterialBuffer(const GpuMaterialBuffer&) = delete;

        GpuMaterialBuffer& operator=(GpuMaterialBuffer&&) = delete;
        GpuMaterialBuffer& operator=(const GpuMaterialBuffer&) = delete;

        ~GpuMaterialBuffer();

        void MarkMaterialsUsed(RenderComponent *);
        void MarkMaterialsUnused(RenderComponent *);

        u32 GetMaterialIndex(const MaterialPtr) const;
        GpuBuffer GetMaterialBuffer() const;

        void UploadDataToGpu();

        static GpuMaterialBufferPtr Create(const usize maxMaterials) {
            return GpuMaterialBufferPtr(new GpuMaterialBuffer(maxMaterials));
        }

    private:
        void CopyMaterialToGpuStaging_(const MaterialPtr& material, const i32 index);

    private:
        GpuTypedBufferPtr<GpuMaterial> materials_;
        // These are the materials we draw from to calculate the material-indices map
        std::unordered_map<MaterialPtr, std::unordered_set<RenderComponent *>> availableMaterials_;
        std::unordered_map<MaterialPtr, std::vector<TextureMemResidencyGuard>> residentTexturesPerMaterial_;
        // Indices can change completely if new materials are added
        std::unordered_map<MaterialPtr, u32> usedIndices_;
        std::unordered_set<MaterialPtr> pendingMaterials_;
    };
}