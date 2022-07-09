#pragma once

#include "GL/gl3w.h"
#include "glm/glm.hpp"
#include <memory>
#include <vector>
#include "StratusCommon.h"

namespace stratus {
    enum class GpuBufferType : int {
        // Holds data for streaming through the graphics pipeline
        PRIMITIVE_BUFFER,
        // Specifically holds indices for describing access patterns into PRIMITIVE_BUFFER
        INDEX_BUFFER
    };

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
        GpuBuffer(GpuBufferType, const void * data, const size_t sizeBytes, const Bitfield usage = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE);
        virtual ~GpuBuffer() = default;

        void EnableAttribute(int32_t attribute, int32_t sizePerElem, GpuStorageType, bool normalized, uint32_t stride, uint32_t offset, uint32_t divisor = 0);
        void Bind() const;
        void Unbind() const;

        // Maps the GPU memory into system memory
        void * MapMemory() const;
        void UnmapMemory() const;
        bool IsMemoryMapped() const;

    private:
        std::shared_ptr<GpuBufferImpl> _impl;
    };

    // Holds different gpu buffers and can bind/unbind them all as a group
    struct GpuArrayBuffer {
        GpuArrayBuffer();
        ~GpuArrayBuffer() = default;

        void AddBuffer(const GpuBuffer&);
        size_t GetNumBuffers() const;
        GpuBuffer& GetBuffer(size_t);
        const GpuBuffer& GetBuffer(size_t) const;
        void UnmapAllMemory() const;
        bool IsMemoryMapped() const;
        void Bind() const;
        void Unbind() const;
        void Clear();

    private:
        std::shared_ptr<std::vector<GpuBuffer>> _buffers;
    };
}