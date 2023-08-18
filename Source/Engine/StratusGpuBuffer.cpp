#include "StratusGpuBuffer.h"
#include <functional>
#include <iostream>
#include <algorithm>
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
        if (type & GPU_MAP_COHERENT) {
            usage |= GL_MAP_COHERENT_BIT;
        }
        return usage;
    }

    static GLenum _ConvertBufferType(i32 type) {
        GpuBindingPoint type_ = static_cast<GpuBindingPoint>(type);
        switch (type_) {
        case GpuBindingPoint::ARRAY_BUFFER: return GL_ARRAY_BUFFER;
        case GpuBindingPoint::ELEMENT_ARRAY_BUFFER: return GL_ELEMENT_ARRAY_BUFFER;
        case GpuBindingPoint::UNIFORM_BUFFER: return GL_UNIFORM_BUFFER;
        case GpuBindingPoint::SHADER_STORAGE_BUFFER: return GL_SHADER_STORAGE_BUFFER;
        case GpuBindingPoint::DRAW_INDIRECT_BUFFER: return GL_DRAW_INDIRECT_BUFFER;
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

    static u32 _CalculateSizeBytes(i32 sizePerElem, GpuStorageType type) {
        switch (type) {
        case GpuStorageType::BYTE:
        case GpuStorageType::UNSIGNED_BYTE: return sizePerElem * sizeof(u8);
        case GpuStorageType::SHORT:
        case GpuStorageType::UNSIGNED_SHORT: return sizePerElem * sizeof(u16);
        case GpuStorageType::INT:
        case GpuStorageType::UNSIGNED_INT: return sizePerElem * sizeof(u32);
        case GpuStorageType::FLOAT: return sizePerElem * sizeof(f32);
        }

        throw std::invalid_argument("Unable to calculate size in bytes");
    }

    static void _CreateBuffer(GLuint & buffer, const void * data, const usize sizeBytes, const Bitfield usage) {
        glCreateBuffers(1, &buffer);
        glNamedBufferStorage(buffer, sizeBytes, data, _ConvertUsageType(usage));
    }

    struct GpuBufferImpl {
        GpuBufferImpl(const void * data, const usize sizeBytes, const Bitfield usage) 
            : _sizeBytes(sizeBytes) {
            _CreateBuffer(_buffer, data, sizeBytes, usage);
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

    void EnableAttribute(i32 attribute, 
                         i32 sizePerElem, 
                         GpuStorageType storage, 
                         bool normalized, 
                         u32 stride, 
                         u32 offset, 
                         u32 divisor = 0) {

        // If we exceed OpenGL's max of 4, we need to calculate a new stride that we
        // can use in the loop below
        if (sizePerElem > 4) {
            // Ex: for a 4x4 f32 matrix this will be 64 (16 * sizeof(f32))
            const u32 totalSizeBytes = _CalculateSizeBytes(sizePerElem, storage);
            stride = stride + totalSizeBytes;
        }
        
        const auto enable = [this, attribute, sizePerElem, storage, normalized, stride, offset, divisor]() {
            // OpenGL caps each attrib to 4 elements, so if we have one that's larger
            // then we need treat it as multiple attribs
            for (i32 i = 0, elem = 0; elem < sizePerElem; ++i, elem += 4) {
                const i32 pos = attribute + i;
                const i32 elemSize = (sizePerElem - elem) > 4 ? 4 : (sizePerElem - elem);
                const u32 totalSizeBytes = _CalculateSizeBytes(elemSize, storage);
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
        glBindBuffer(_ConvertBufferType(i32(point)), _buffer);
        for (auto& enable : _enableAttributes) enable();
    }

    void Unbind(const GpuBindingPoint point) const {
        glBindBuffer(_ConvertBufferType(i32(point)), 0);
    }

    void BindBase(const GpuBaseBindingPoint point, const u32 index) const {
        glBindBufferBase(_ConvertBufferType(i32(point)), index, _buffer);
    }

    void * MapMemory(const Bitfield access) const {
        return MapMemory(access, 0, _sizeBytes);
    }

    void * MapMemory(const Bitfield access, isize offset, usize length) const {
        _isMemoryMapped = true;
        void * ptr = glMapNamedBufferRange(_buffer, offset, length, _ConvertUsageType(access));
        return ptr;
    }
    
    void UnmapMemory() const {
        glUnmapNamedBuffer(_buffer);
        _isMemoryMapped = false;
    }

    bool IsMemoryMapped() const {
        return _isMemoryMapped;
    }

    usize SizeBytes() const {
        return _sizeBytes;
    }

    void CopyDataToBuffer(isize offset, usize size, const void * data) {
        if (offset + size > SizeBytes()) {
            throw std::runtime_error("offset+size exceeded maximum GPU buffer size");
        }
        glNamedBufferSubData(_buffer, offset, size, data);
    }

    void CopyDataFromBuffer(const GpuBufferImpl& buffer) {
        if (SizeBytes() < buffer.SizeBytes()) {
            throw std::runtime_error("Attempt to copy larger buffer to smaller buffer");
        }
        if (this == &buffer) {
            throw std::runtime_error("Attempt to copy from buffer to itself");
        }
        glCopyNamedBufferSubData(buffer._buffer, _buffer, 0, 0, buffer.SizeBytes());
    }

    void CopyDataFromBufferToSysMem(isize offset, usize size, void * data) {
        if (offset + size > SizeBytes()) {
            throw std::runtime_error("offset+size exceeded maximum GPU buffer size");
        }
        glGetNamedBufferSubData(_buffer, offset, size, data);
    }

    void FinalizeMemory() {
        /*
        VV does not work
        const void * data = MapMemory();
        GLuint old = _buffer;
        const Bitfield usage = 0;
        _buffer = 0;
        _CreateBuffer(_buffer, data, SizeBytes(), usage);
        glDeleteBuffers(1, &old);
        */
    }

    private:
        GLuint _buffer;
        usize _sizeBytes;
        mutable bool _isMemoryMapped = false;

        std::vector<GpuBufferCommand> _enableAttributes;
    };

    GpuBuffer::GpuBuffer(const void * data, const usize sizeBytes, const Bitfield usage)
        : impl_(std::make_shared<GpuBufferImpl>(data, sizeBytes, usage)) {}

    void GpuBuffer::EnableAttribute(i32 attribute, i32 sizePerElem, GpuStorageType storage, bool normalized, u32 stride, u32 offset, u32 divisor) {
        impl_->EnableAttribute(attribute, sizePerElem, storage, normalized, stride, offset, divisor);
    }

    void GpuBuffer::Bind(const GpuBindingPoint point) const {
        impl_->Bind(point);
    }

    void GpuBuffer::Unbind(const GpuBindingPoint point) const {
        impl_->Unbind(point);
    }

    void GpuBuffer::BindBase(const GpuBaseBindingPoint point, const u32 index) const {
        impl_->BindBase(point, index);
    }

    void * GpuBuffer::MapMemory(const Bitfield access) const {
        return impl_->MapMemory(access);
    }

    void * GpuBuffer::MapMemory(const Bitfield access, isize offset, usize length) const {
        return impl_->MapMemory(access, offset, length);
    }

    void GpuBuffer::UnmapMemory() const {
        impl_->UnmapMemory();
    }

    bool GpuBuffer::IsMemoryMapped() const {
        return impl_->IsMemoryMapped();
    }

    usize GpuBuffer::SizeBytes() const {
        return impl_->SizeBytes();
    }

    void GpuBuffer::CopyDataToBuffer(isize offset, usize size, const void * data) {
        impl_->CopyDataToBuffer(offset, size, data);
    }

    void GpuBuffer::CopyDataFromBuffer(const GpuBuffer& buffer) {
        if (impl_ == nullptr || buffer.impl_ == nullptr) {
            throw std::runtime_error("Attempt to use null GpuBuffer");
        }
        impl_->CopyDataFromBuffer(*buffer.impl_);
    }

    void GpuBuffer::CopyDataFromBufferToSysMem(isize offset, usize size, void * data) {
        impl_->CopyDataFromBufferToSysMem(offset, size, data);
    }

    void GpuBuffer::FinalizeMemory() {
        impl_->FinalizeMemory();
    }

    GpuPrimitiveBuffer::GpuPrimitiveBuffer(const GpuPrimitiveBindingPoint type, const void * data, const usize sizeBytes, const Bitfield usage)
        : GpuBuffer(data, sizeBytes, usage),
          type_(type) {}

    void GpuPrimitiveBuffer::Bind() const {
        GpuBuffer::Bind(static_cast<GpuBindingPoint>(type_));
    }

    void GpuPrimitiveBuffer::Unbind() const {
        GpuBuffer::Unbind(static_cast<GpuBindingPoint>(type_));
    }

    GpuArrayBuffer::GpuArrayBuffer()
        : buffers_(std::make_shared<std::vector<std::unique_ptr<GpuPrimitiveBuffer>>>()) {}

    void GpuArrayBuffer::AddBuffer(const GpuPrimitiveBuffer& buffer) {
        buffers_->push_back(std::make_unique<GpuPrimitiveBuffer>(buffer));
    }

    void GpuArrayBuffer::Bind() const {
        for (auto& buffer : *buffers_) buffer->Bind();
    }

    void GpuArrayBuffer::Unbind() const {
        for (auto& buffer : *buffers_) buffer->Unbind();
    }

    void GpuArrayBuffer::Clear() {
        buffers_->clear();
    }

    usize GpuArrayBuffer::GetNumBuffers() const {
        return buffers_->size();
    }

    GpuPrimitiveBuffer& GpuArrayBuffer::GetBuffer(usize index) {
        return *(*buffers_)[index];
    }

    const GpuPrimitiveBuffer& GpuArrayBuffer::GetBuffer(usize index) const {
        return *(*buffers_)[index];
    }

    void GpuArrayBuffer::UnmapAllMemory() const {
        for (auto& buffer : *buffers_) buffer->UnmapMemory();
    }

    bool GpuArrayBuffer::IsMemoryMapped() const {
        for (auto& buffer : *buffers_) {
            if (buffer->IsMemoryMapped()) return true;
        }
        return false;
    }

    void GpuArrayBuffer::FinalizeAllMemory() {
        for (auto& buffer : *buffers_) {
            buffer->FinalizeMemory();
        }
    }

    // Responsible for allocating vertex and index data. All data is stored
    // in two giant GPU buffers (one for vertices, one for indices).
    //
    // This is NOT thread safe as only the main thread should be using it since 
    // it performs GPU memory allocation.
    //
    // It can support a maximum of UINT_MAX vertices and UINT_MAX indices.
    // class GpuMeshAllocator final {
    //     struct _MeshData {
    //         usize nextByte;
    //         usize lastByte;
    //     };

    //     GpuMeshAllocator() {}

    // public:
    //     // Allocates 64-byte block vertex data where each element represents a GpuMeshData type.
    //     //
    //     // @return offset into global GPU vertex data array where data begins
    //     static u32 AllocateVertexData(const u32 numVertices);
    //     // @return offset into global GPU index data array where data begins
    //     static u32 AllocateIndexData(const u32 numIndices);

    //     // Deallocation
    //     static void DeallocateVertexData(const u32 offset, const u32 numVertices);
    //     static void DeallocateIndexData(const u32 offset, const u32 numIndices);

    //     static void CopyVertexData(const std::vector<GpuMeshData>&, const u32 offset);
    //     static void CopyIndexData(const std::vector<u32>&, const u32 offset);

    //     static void BindBase(const GpuBaseBindingPoint&);

    // private:
    //     vo
    //     void _Shutdown();

    // private:
    //     static GpuBuffer _vertices;
    //     static GpuBuffer _indices;
    //     static _MeshData _lastVertex;
    //     static _MeshData _lastIndex;
    //     static bool _initialized;
    // };

    GpuBuffer GpuMeshAllocator::vertices_;
    GpuBuffer GpuMeshAllocator::indices_;
    GpuMeshAllocator::_MeshData GpuMeshAllocator::lastVertex_;
    GpuMeshAllocator::_MeshData GpuMeshAllocator::lastIndex_;
    std::vector<GpuMeshAllocator::_MeshData> GpuMeshAllocator::freeVertices_;
    std::vector<GpuMeshAllocator::_MeshData> GpuMeshAllocator::freeIndices_;
    bool GpuMeshAllocator::initialized_ = false;
    static constexpr usize startVertices = 1024 * 1024 * 10;
    static constexpr usize minVerticesPerAlloc = startVertices; //1024 * 1024;
    static constexpr usize maxVertexBytes = std::numeric_limits<u32>::max() * sizeof(GpuMeshData);
    static constexpr usize maxIndexBytes = std::numeric_limits<u32>::max() * sizeof(u32);
    //static constexpr usize maxVertexBytes = startVertices * sizeof(GpuMeshData);
    //static constexpr usize maxIndexBytes = startVertices * sizeof(u32);

    GpuMeshAllocator::_MeshData * GpuMeshAllocator::FindFreeSlot_(std::vector<GpuMeshAllocator::_MeshData>& freeList, const usize bytes) {
        for (_MeshData& data : freeList) {
            auto remaining = RemainingBytes_(data);
            // TODO: Instead choose smallest buffer that will work rather than first available?
            if (remaining >= bytes) {
                return &data;
            }
        }
        return nullptr;
    }

    u32 GpuMeshAllocator::AllocateData_(const u32 size, const usize byteMultiplier, const usize maxBytes, 
                                             GpuBuffer& buffer, _MeshData& data, std::vector<GpuMeshAllocator::_MeshData>& freeList) {
        assert(size > 0);

        _MeshData * dataPtr = &data;
        const usize totalSizeBytes = usize(size) * byteMultiplier;
        const usize remainingBytes = RemainingBytes_(data);

        if (totalSizeBytes > remainingBytes) {
            // See if one of the slots has data we can use
            _MeshData * freeSlot = FindFreeSlot_(freeList, totalSizeBytes);
            if (freeSlot) {
                dataPtr = freeSlot;
            }
            // If not perform a resize
            else {
                const usize newSizeBytes = data.lastByte + std::max(usize(size), minVerticesPerAlloc) * byteMultiplier;
                if (newSizeBytes > maxBytes) {
                    throw std::runtime_error("Maximum GpuMesh bytes exceeded");
                }
                Resize_(buffer, data, newSizeBytes);
            }
        }

        const u32 offset = dataPtr->nextByte / byteMultiplier;
        dataPtr->nextByte += totalSizeBytes;

        // We pulled from a free list - delete if empty
        if (dataPtr != &data && RemainingBytes_(*dataPtr) == 0) {
            std::stable_partition(freeList.begin(), freeList.end(),
                [dataPtr](_MeshData& d) { return &d != dataPtr; }
            );
            freeList.pop_back();
        }

        return offset;
    }

    u32 GpuMeshAllocator::AllocateVertexData(const u32 numVertices) {
        return AllocateData_(numVertices, sizeof(GpuMeshData), maxVertexBytes, vertices_, lastVertex_, freeVertices_);
    }

    u32 GpuMeshAllocator::AllocateIndexData(const u32 numIndices) {
        return AllocateData_(numIndices, sizeof(u32), maxIndexBytes, indices_, lastIndex_, freeIndices_);
    }

    void GpuMeshAllocator::DeallocateData_(_MeshData& last, std::vector<GpuMeshAllocator::_MeshData>& freeList, const usize offsetBytes, const usize lastByte) {
        // If it's at the end just return it to the last vertex to allow O(1) allocation
        if (lastByte == last.lastByte) {
            last.nextByte = offsetBytes;
        }
        else {
            // Find an appropriate slot (either existing or new) and merge slots that are close by
            _MeshData data;
            data.nextByte = offsetBytes;
            data.lastByte = lastByte;
            bool done = false;
            for (_MeshData& entry : freeList) {
                // Overlapping entries - merge
                if (offsetBytes <= entry.lastByte) {
                    entry.lastByte = lastByte;
                    done = true;
                    break;
                }
            }

            if (!done) {
                freeList.push_back(data);

                const auto comparison = [](const _MeshData& left, const _MeshData& right) {
                    return left.lastByte < right.lastByte;
                };

                std::sort(freeList.begin(), freeList.end(), comparison);
            }

            // Try to merge any that we can
            if (freeList.size() > 1) {
                usize numElements = 1;
                for (usize i = 0; i < freeList.size() - 1; ++i) {
                    _MeshData& current = freeList[i];
                    _MeshData& next = freeList[i + 1];
                    // See if current and next can be merged
                    if (next.nextByte <= current.lastByte) {
                        next.nextByte = current.nextByte;
                        current.nextByte = 0;
                        current.lastByte = 0;
                    }
                    else {
                        ++numElements;
                    }
                }

                // Clear out dead entries
                if (numElements != freeList.size()) {
                    auto it = std::stable_partition(freeList.begin(), freeList.end(),
                        [](const _MeshData& d) { return RemainingBytes_(d) != 0; }
                    );
                    auto removed = std::distance(it, freeList.end());
                    for (i32 i = 0; i < removed; ++i) {
                        freeList.pop_back();
                    }
                }
            }

            // See if we can merge it back into the main list
            if (freeList.size() == 1) {
                if (last.nextByte <= freeList[0].lastByte) {
                    last.nextByte = freeList[0].nextByte;
                    freeList.clear();
                }
            }
        }
    }

    void GpuMeshAllocator::DeallocateVertexData(const u32 offset, const u32 numVertices) {
        const usize offsetBytes = offset * sizeof(GpuMeshData);
        const usize lastByte = offsetBytes + numVertices * sizeof(GpuMeshData);
        DeallocateData_(lastVertex_, freeVertices_, offsetBytes, lastByte);
    }

    void GpuMeshAllocator::DeallocateIndexData(const u32 offset, const u32 numIndices) {
        const usize offsetBytes = offset * sizeof(u32);
        const usize lastByte = offsetBytes + numIndices * sizeof(u32);
        DeallocateData_(lastIndex_, freeIndices_, offsetBytes, lastByte);
    }

    void GpuMeshAllocator::CopyVertexData(const std::vector<GpuMeshData>& data, const u32 offset) {
        const isize byteOffset = isize(offset) * sizeof(GpuMeshData);
        vertices_.CopyDataToBuffer(byteOffset, data.size() * sizeof(GpuMeshData), (const void *)data.data());
    }

    void GpuMeshAllocator::CopyIndexData(const std::vector<u32>& data, const u32 offset) {
        const isize byteOffset = isize(offset) * sizeof(u32);
        indices_.CopyDataToBuffer(byteOffset, data.size() * sizeof(u32), (const void *)data.data());
    }

    void GpuMeshAllocator::BindBase(const GpuBaseBindingPoint& point, const u32 index) {
        vertices_.BindBase(point, index);
    }

    void GpuMeshAllocator::BindElementArrayBuffer() {
        indices_.Bind(GpuBindingPoint::ELEMENT_ARRAY_BUFFER);
    }

    void GpuMeshAllocator::UnbindElementArrayBuffer() {
        indices_.Unbind(GpuBindingPoint::ELEMENT_ARRAY_BUFFER);
    }

    void GpuMeshAllocator::Initialize_() {
        if (initialized_) return;
        initialized_ = true;
        lastVertex_.nextByte = 0;
        lastIndex_.nextByte = 0;
        Resize_(vertices_, lastVertex_, startVertices * sizeof(GpuMeshData));
        Resize_(indices_, lastIndex_, startVertices * sizeof(u32));
    }

    void GpuMeshAllocator::Shutdown_() {
        vertices_ = GpuBuffer();
        indices_ = GpuBuffer();
        initialized_ = false;
    }

    void GpuMeshAllocator::Resize_(GpuBuffer& buffer, _MeshData& data, const usize newSizeBytes) {
        STRATUS_LOG << "Resizing: " << newSizeBytes << std::endl;
        GpuBuffer resized = GpuBuffer(nullptr, newSizeBytes, GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE);
        // Null check
        if (buffer != GpuBuffer()) {
            resized.CopyDataFromBuffer(buffer);
        }
        data.lastByte = newSizeBytes;
        buffer = resized;
    }

    usize GpuMeshAllocator::RemainingBytes_(const _MeshData& data) {
        return data.lastByte - data.nextByte;
    }

    u32 GpuMeshAllocator::FreeVertices() {
        u32 vertices = static_cast<u32>(RemainingBytes_(lastVertex_) / sizeof(GpuMeshData));
        for (auto& data : freeVertices_) {
            vertices += static_cast<u32>(RemainingBytes_(data) / sizeof(GpuMeshData));
        }
        return vertices;
    }

    u32 GpuMeshAllocator::FreeIndices() {
        u32 indices = static_cast<u32>(RemainingBytes_(lastIndex_) / sizeof(u32));
        for (auto& data : freeIndices_) {
            indices += static_cast<u32>(RemainingBytes_(data) / sizeof(u32));
        }
        return indices;
    }
}