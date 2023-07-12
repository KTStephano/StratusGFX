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

namespace stratus {
    struct GpuMaterialBuffer;
    typedef std::shared_ptr<GpuMaterialBuffer> GpuMaterialBufferPtr;

    struct GpuMaterialBuffer {
        GpuMaterialBuffer(size_t maxMaterials);
        
        GpuMaterialBuffer(GpuMaterialbuffer&&) = default;
        GpuMaterialBuffer(const GpuMaterialBuffer&) = delete;

        GpuMaterialBuffer& operator=(GpuMaterialBuffer&&) = delete;
        GpuMaterialBuffer& operator=(const GpuMaterialBuffer&) = delete;

        ~GpuMaterialBuffer();
    };
}