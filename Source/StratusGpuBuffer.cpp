#include "StratusGpuBuffer.h"
#include <functional>
#include <iostream>
#include "StratusApplicationThread.h"
#include "StratusLog.h"

namespace stratus {
    typedef std::function<void(void)> GpuBufferCommand;

    static GLbitfield _ConvertUsageType(Bitfield type) {
        GLbitfield usage = 0;
        if (type & GPU_DYNAMIC_DATA) {
            usage |= GL_DYNAMIC_STORAGE_BIT;
        }
        if (type & GPU_MAP_READ) {
            usage |= GL_MAP_READ_BIT;
        }
        if (type & GPU_MAP_WRITE) {
            usage |= GL_MAP_WRITE_BIT;
        }
        if (type & GPU_MAP_PERSISTENT) {
            usage |= GL_MAP_PERSISTENT_BIT;
        }
        return usage;
    }

    static GLenum _ConvertBufferType(int type) {
        GpuBindingPoint type_ = static_cast<GpuBindingPoint>(type);
        switch (type_) {
        case GpuBindingPoint::ARRAY_BUFFER: return GL_ARRAY_BUFFER;
        case GpuBindingPoint::ELEMENT_ARRAY_BUFFER: return GL_ELEMENT_ARRAY_BUFFER;
        case GpuBindingPoint::UNIFORM_BUFFER: return GL_UNIFORM_BUFFER;
        case GpuBindingPoint::SHADER_STORAGE_BUFFER: return GL_SHADER_STORAGE_BUFFER;
        }

        throw std::invalid_argument("Unknown buffer type");
    }

    static GLenum _ConvertStorageType(GpuStorageType type) {
        switch (type) {
        case GpuStorageType::BYTE: return GL_BYTE;
        case GpuStorageType::UNSIGNED_BYTE: return GL_UNSIGNED_BYTE;
        case GpuStorageType::SHORT: return GL_SHORT;
        case GpuStorageType::UNSIGNED_SHORT: return GL_UNSIGNED_SHORT;
        case GpuStorageType::INT: return GL_INT;
        case GpuStorageType::UNSIGNED_INT: return GL_UNSIGNED_INT;
        case GpuStorageType::FLOAT: return GL_FLOAT;
        }

        throw std::invalid_argument("Unknown storage type");
    }

    static uint32_t _CalculateSizeBytes(int32_t sizePerElem, GpuStorageType type) {
        switch (type) {
        case GpuStorageType::BYTE:
        case GpuStorageType::UNSIGNED_BYTE: return sizePerElem * sizeof(uint8_t);
        case GpuStorageType::SHORT:
        case GpuStorageType::UNSIGNED_SHORT: return sizePerElem * sizeof(uint16_t);
        case GpuStorageType::INT:
        case GpuStorageType::UNSIGNED_INT: return sizePerElem * sizeof(uint32_t);
        case GpuStorageType::FLOAT: return sizePerElem * sizeof(float);
        }

        throw std::invalid_argument("Unable to calculate size in bytes");
    }

    struct GpuBufferImpl {
        GpuBufferImpl(const void * data, const uintptr_t sizeBytes, const Bitfield usage) 
            : _sizeBytes(sizeBytes) {

            glCreateBuffers(1, &_buffer);
            glNamedBufferStorage(_buffer, sizeBytes, data, _ConvertUsageType(usage));
        }

        ~GpuBufferImpl() {
            if (ApplicationThread::Instance()->CurrentIsApplicationThread()) {
                glDeleteBuffers(1, &_buffer);
            }
            else {
                auto buffer = _buffer;
                ApplicationThread::Instance()->Queue([buffer]() { GLuint buf = buffer; glDeleteBuffers(1, &buf); });
            }
        }

    void EnableAttribute(int32_t attribute, 
                         int32_t sizePerElem, 
                         GpuStorageType storage, 
                         bool normalized, 
                         uint32_t stride, 
                         uint32_t offset, 
                         uint32_t divisor = 0) {

        // If we exceed OpenGL's max of 4, we need to calculate a new stride that we
        // can use in the loop below
        if (sizePerElem > 4) {
            // Ex: for a 4x4 float matrix this will be 64 (16 * sizeof(float))
            const uint32_t totalSizeBytes = _CalculateSizeBytes(sizePerElem, storage);
            stride = stride + totalSizeBytes;
        }
        
        const auto enable = [this, attribute, sizePerElem, storage, normalized, stride, offset, divisor]() {
            // OpenGL caps each attrib to 4 elements, so if we have one that's larger
            // then we need treat it as multiple attribs
            for (int32_t i = 0, elem = 0; elem < sizePerElem; ++i, elem += 4) {
                const int32_t pos = attribute + i;
                const int32_t elemSize = (sizePerElem - elem) > 4 ? 4 : (sizePerElem - elem);
                const uint32_t totalSizeBytes = _CalculateSizeBytes(elemSize, storage);
                glEnableVertexAttribArray(pos);
                glVertexAttribPointer(
                    pos, 
                    elemSize,
                    _ConvertStorageType(storage),
                    normalized ? GL_TRUE : GL_FALSE,
                    stride, // Offset from one element to the next
                    (void *)(offset + i * totalSizeBytes) // one-time offset before reading first element
                );

                // If 0, data increments by 1 for each vertex
                glVertexAttribDivisor(pos, divisor);
            }
        };

        _enableAttributes.push_back(enable);
    }

    void Bind(const GpuBindingPoint point) const {
        glBindBuffer(_ConvertBufferType(int(point)), _buffer);
        for (auto& enable : _enableAttributes) enable();
    }

    void Unbind(const GpuBindingPoint point) const {
        glBindBuffer(_ConvertBufferType(int(point)), 0);
    }

    void BindBase(const GpuBaseBindingPoint point, const uint32_t index) {
        glBindBufferBase(_ConvertBufferType(int(point)), index, _buffer);
    }

    void * MapMemory() const {
        _isMemoryMapped = true;
        void * ptr = glMapNamedBuffer(_buffer, GL_READ_WRITE);
        return ptr;
    }
    
    void UnmapMemory() const {
        glUnmapNamedBuffer(_buffer);
        _isMemoryMapped = false;
    }

    bool IsMemoryMapped() const {
        return _isMemoryMapped;
    }

    uintptr_t SizeBytes() const {
        return _sizeBytes;
    }

    void CopyDataToBuffer(intptr_t offset, uintptr_t size, const void * data) {
        if (offset + size > SizeBytes()) {
            throw std::runtime_error("offset+size exceeded maximum GPU buffer size");
        }
        glNamedBufferSubData(_buffer, offset, size, data);
    }

    private:
        GLuint _buffer;
        uintptr_t _sizeBytes;
        mutable bool _isMemoryMapped = false;

        std::vector<GpuBufferCommand> _enableAttributes;
    };

    GpuBuffer::GpuBuffer(const void * data, const uintptr_t sizeBytes, const Bitfield usage)
        : _impl(std::make_shared<GpuBufferImpl>(data, sizeBytes, usage)) {}

    void GpuBuffer::EnableAttribute(int32_t attribute, int32_t sizePerElem, GpuStorageType storage, bool normalized, uint32_t stride, uint32_t offset, uint32_t divisor) {
        _impl->EnableAttribute(attribute, sizePerElem, storage, normalized, stride, offset, divisor);
    }

    void GpuBuffer::Bind(const GpuBindingPoint point) const {
        _impl->Bind(point);
    }

    void GpuBuffer::Unbind(const GpuBindingPoint point) const {
        _impl->Unbind(point);
    }

    void GpuBuffer::BindBase(const GpuBaseBindingPoint point, const uint32_t index) {
        _impl->BindBase(point, index);
    }

    void * GpuBuffer::MapMemory() const {
        return _impl->MapMemory();
    }

    void GpuBuffer::UnmapMemory() const {
        _impl->UnmapMemory();
    }

    bool GpuBuffer::IsMemoryMapped() const {
        return _impl->IsMemoryMapped();
    }

    uintptr_t GpuBuffer::SizeBytes() const {
        return _impl->SizeBytes();
    }

    void GpuBuffer::CopyDataToBuffer(intptr_t offset, uintptr_t size, const void * data) {
        _impl->CopyDataToBuffer(offset, size, data);
    }

    GpuPrimitiveBuffer::GpuPrimitiveBuffer(const GpuPrimitiveBindingPoint type, const void * data, const uintptr_t sizeBytes, const Bitfield usage)
        : GpuBuffer(data, sizeBytes, usage),
          _type(type) {}

    void GpuPrimitiveBuffer::Bind() const {
        GpuBuffer::Bind(static_cast<GpuBindingPoint>(_type));
    }

    void GpuPrimitiveBuffer::Unbind() const {
        GpuBuffer::Unbind(static_cast<GpuBindingPoint>(_type));
    }

    GpuArrayBuffer::GpuArrayBuffer()
        : _buffers(std::make_shared<std::vector<std::unique_ptr<GpuPrimitiveBuffer>>>()) {}

    void GpuArrayBuffer::AddBuffer(const GpuPrimitiveBuffer& buffer) {
        _buffers->push_back(std::make_unique<GpuPrimitiveBuffer>(buffer));
    }

    void GpuArrayBuffer::Bind() const {
        for (auto& buffer : *_buffers) buffer->Bind();
    }

    void GpuArrayBuffer::Unbind() const {
        for (auto& buffer : *_buffers) buffer->Unbind();
    }

    void GpuArrayBuffer::Clear() {
        _buffers->clear();
    }

    size_t GpuArrayBuffer::GetNumBuffers() const {
        return _buffers->size();
    }

    GpuPrimitiveBuffer& GpuArrayBuffer::GetBuffer(size_t index) {
        return *(*_buffers)[index];
    }

    const GpuPrimitiveBuffer& GpuArrayBuffer::GetBuffer(size_t index) const {
        return *(*_buffers)[index];
    }

    void GpuArrayBuffer::UnmapAllMemory() const {
        for (auto& buffer : *_buffers) buffer->UnmapMemory();
    }

    bool GpuArrayBuffer::IsMemoryMapped() const {
        for (auto& buffer : *_buffers) {
            if (buffer->IsMemoryMapped()) return true;
        }
        return false;
    }
}