#pragma once

#include "GL/gl3w.h"
#include "glm/glm.hpp"
#include <memory>
#include <vector>

namespace stratus {
    enum class GpuBufferType : int {
        // Holds data for streaming through the graphics pipeline
        PRIMITIVE_BUFFER,
        // Specifically holds indices for describing access patterns into PRIMITIVE_BUFFER
        INDEX_BUFFER
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

    struct GpuBufferImpl;
    struct GpuArrayBufferImpl;

    // A gpu buffer holds primitive data usually in the form of floats, ints and shorts
    // TODO: Look into use cases for things other than STATIC_DRAW
    struct GpuBuffer {
        GpuBuffer() {}
        GpuBuffer(GpuBufferType, const void * data, const size_t sizeBytes);
        ~GpuBuffer() = default;

        void EnableAttribute(int32_t attribute, int32_t sizePerElem, GpuStorageType, bool normalized, uint32_t stride, uint32_t offset, uint32_t divisor = 0);
        void Bind() const;
        void Unbind() const;

        void * MapReadWrite() const;
        void UnmapReadWrite() const;
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
        void UnmapAllReadWrite() const;
        bool IsMemoryMapped() const;
        void Bind() const;
        void Unbind() const;
        void Clear();

    private:
        std::shared_ptr<std::vector<GpuBuffer>> _buffers;
    };
}