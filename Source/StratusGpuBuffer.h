#pragma once

#include "GL/gl3w.h"
#include "glm/glm.hpp"
#include <memory>
#include <vector>
#include <cstdint>
#include "StratusCommon.h"

namespace stratus {
    enum class GpuBindingPoint : int {
        // Good for things like vertices and normals
        ARRAY_BUFFER            = BITMASK64_POW2(1),
        // Good for indices
        ELEMENT_ARRAY_BUFFER    = BITMASK64_POW2(2),
        // Allows read-only uniform buffer access
        UNIFORM_BUFFER          = BITMASK64_POW2(3),
        // Allows read and write shader buffer access
        SHADER_STORAGE_BUFFER   = BITMASK64_POW2(4)
    };

    // A more restrictive set of bindings good for things like floating point (vertex, normal, etc.)
    // and integer (index) buffers
    enum class GpuPrimitiveBindingPoint : int {
        ARRAY_BUFFER            = int(GpuBindingPoint::ARRAY_BUFFER),
        ELEMENT_ARRAY_BUFFER    = int(GpuBindingPoint::ELEMENT_ARRAY_BUFFER)
    };

    enum class GpuBaseBindingPoint : int {
        UNIFORM_BUFFER          = int(GpuBindingPoint::UNIFORM_BUFFER),
        SHADER_STORAGE_BUFFER   = int(GpuBindingPoint::SHADER_STORAGE_BUFFER)
    };

    enum class GpuStorageType : int {
        BYTE,
        UNSIGNED_BYTE,
        SHORT,
        UNSIGNED_SHORT,
        INT,
        UNSIGNED_INT,
        FLOAT
    };

    typedef int Bitfield;

    // Describes how the data will likely be used, meaning whether it will be changed
    // frequently, mapped for read/write or mapped persistently

    // Data will be set at creation and never touched again
    constexpr Bitfield GPU_STATIC_DATA = BITMASK64_POW2(0);
    // Data will be copied directly through the copy API often
    constexpr Bitfield GPU_DYNAMIC_DATA = BITMASK64_POW2(1);
    // Memory will be mapped for reading
    constexpr Bitfield GPU_MAP_READ = BITMASK64_POW2(2);
    // Memory will be mapped for writing outside of the API copy calls
    constexpr Bitfield GPU_MAP_WRITE = BITMASK64_POW2(3);
    // Memory will be mapped and continuously read from and written to without unmapping
    constexpr Bitfield GPU_MAP_PERSISTENT = BITMASK64_POW2(4);

    struct GpuBufferImpl;
    struct GpuArrayBufferImpl;

    // A gpu buffer holds primitive data usually in the form of floats, ints and shorts
    // TODO: Look into use cases for things other than STATIC_DRAW
    struct GpuBuffer {
        GpuBuffer() {}
        GpuBuffer(const void * data, const uintptr_t sizeBytes, const Bitfield usage = GPU_MAP_READ | GPU_MAP_WRITE);
        virtual ~GpuBuffer() = default;

        void EnableAttribute(int32_t attribute, int32_t sizePerElem, GpuStorageType, bool normalized, uint32_t stride, uint32_t offset, uint32_t divisor = 0);
        virtual void Bind(const GpuBindingPoint) const;
        virtual void Unbind(const GpuBindingPoint) const;
        // From what I can tell there shouldn't be a need to unbind UBOs
        virtual void BindBase(const GpuBaseBindingPoint, const uint32_t index);

        // Maps the GPU memory into system memory - make sure READ, WRITE, or PERSISTENT mapping is enabled
        void * MapMemory() const;
        void UnmapMemory() const;
        bool IsMemoryMapped() const;

        uintptr_t SizeBytes() const;
        // Make sure GPU_DYNAMIC_DATA is set
        void CopyDataToBuffer(intptr_t offset, uintptr_t size, const void * data);

        // Memory mapping and data copying won't work after this
        void FinalizeMemory();

    protected:
        std::shared_ptr<GpuBufferImpl> _impl;
    };

    struct GpuPrimitiveBuffer final : public GpuBuffer {
        GpuPrimitiveBuffer() : GpuBuffer() {}
        GpuPrimitiveBuffer(const GpuPrimitiveBindingPoint type, const void * data, const uintptr_t sizeBytes, const Bitfield usage = 0);
        virtual ~GpuPrimitiveBuffer() = default;

        void Bind() const;
        void Unbind() const;

    private:
        GpuPrimitiveBindingPoint _type;
    };

    // Holds different gpu buffers and can bind/unbind them all as a group
    struct GpuArrayBuffer final {
        GpuArrayBuffer();
        ~GpuArrayBuffer() = default;

        void AddBuffer(const GpuPrimitiveBuffer&);
        size_t GetNumBuffers() const;
        GpuPrimitiveBuffer& GetBuffer(size_t);
        const GpuPrimitiveBuffer& GetBuffer(size_t) const;
        void UnmapAllMemory() const;
        bool IsMemoryMapped() const;
        void FinalizeAllMemory();
        void Bind() const;
        void Unbind() const;
        void Clear();

    private:
        std::shared_ptr<std::vector<std::unique_ptr<GpuPrimitiveBuffer>>> _buffers;
    };
}