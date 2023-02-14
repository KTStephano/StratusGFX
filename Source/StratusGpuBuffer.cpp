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

    static GLenum _ConvertBufferType(int type) {
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

    static void _CreateBuffer(GLuint & buffer, const void * data, const uintptr_t sizeBytes, const Bitfield usage) {
        glCreateBuffers(1, &buffer);
        glNamedBufferStorage(buffer, sizeBytes, data, _ConvertUsageType(usage));
    }

    struct GpuBufferImpl {
        GpuBufferImpl(const void * data, const uintptr_t sizeBytes, const Bitfield usage) 
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

    void BindBase(const GpuBaseBindingPoint point, const uint32_t index) const {
        glBindBufferBase(_ConvertBufferType(int(point)), index, _buffer);
    }

    void * MapMemory(const Bitfield access) const {
        _isMemoryMapped = true;
        void * ptr = glMapNamedBufferRange(_buffer, 0, _sizeBytes, _ConvertUsageType(access));
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

    void CopyDataFromBuffer(const GpuBufferImpl& buffer) {
        if (SizeBytes() < buffer.SizeBytes()) {
            throw std::runtime_error("Attempt to copy larger buffer to smaller buffer");
        }
        if (this == &buffer) {
            throw std::runtime_error("Attempt to copy from buffer to itself");
        }
        glCopyNamedBufferSubData(buffer._buffer, _buffer, 0, 0, buffer.SizeBytes());
    }

    void CopyDataFromBufferToSysMem(intptr_t offset, uintptr_t size, void * data) {
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

    void GpuBuffer::BindBase(const GpuBaseBindingPoint point, const uint32_t index) const {
        _impl->BindBase(point, index);
    }

    void * GpuBuffer::MapMemory(const Bitfield access) const {
        return _impl->MapMemory(access);
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

    void GpuBuffer::CopyDataFromBuffer(const GpuBuffer& buffer) {
        if (_impl == nullptr || buffer._impl == nullptr) {
            throw std::runtime_error("Attempt to use null GpuBuffer");
        }
        _impl->CopyDataFromBuffer(*buffer._impl);
    }

    void GpuBuffer::CopyDataFromBufferToSysMem(intptr_t offset, uintptr_t size, void * data) {
        _impl->CopyDataFromBufferToSysMem(offset, size, data);
    }

    void GpuBuffer::FinalizeMemory() {
        _impl->FinalizeMemory();
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

    void GpuArrayBuffer::FinalizeAllMemory() {
        for (auto& buffer : *_buffers) {
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
    //         size_t nextByte;
    //         size_t lastByte;
    //     };

    //     GpuMeshAllocator() {}

    // public:
    //     // Allocates 64-byte block vertex data where each element represents a GpuMeshData type.
    //     //
    //     // @return offset into global GPU vertex data array where data begins
    //     static uint32_t AllocateVertexData(const uint32_t numVertices);
    //     // @return offset into global GPU index data array where data begins
    //     static uint32_t AllocateIndexData(const uint32_t numIndices);

    //     // Deallocation
    //     static void DeallocateVertexData(const uint32_t offset, const uint32_t numVertices);
    //     static void DeallocateIndexData(const uint32_t offset, const uint32_t numIndices);

    //     static void CopyVertexData(const std::vector<GpuMeshData>&, const uint32_t offset);
    //     static void CopyIndexData(const std::vector<uint32_t>&, const uint32_t offset);

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

    GpuBuffer GpuMeshAllocator::_vertices;
    GpuBuffer GpuMeshAllocator::_indices;
    GpuMeshAllocator::_MeshData GpuMeshAllocator::_lastVertex;
    GpuMeshAllocator::_MeshData GpuMeshAllocator::_lastIndex;
    std::vector<GpuMeshAllocator::_MeshData> GpuMeshAllocator::_freeVertices;
    std::vector<GpuMeshAllocator::_MeshData> GpuMeshAllocator::_freeIndices;
    bool GpuMeshAllocator::_initialized = false;
    static constexpr size_t minVertices = 65536; //262144;
    static constexpr size_t maxVertexBytes = std::numeric_limits<uint32_t>::max() * sizeof(GpuMeshData);
    static constexpr size_t maxIndexBytes = std::numeric_limits<uint32_t>::max() * sizeof(uint32_t);

    GpuMeshAllocator::_MeshData * GpuMeshAllocator::_FindFreeSlot(std::vector<GpuMeshAllocator::_MeshData>& freeList, const size_t bytes) {
        for (_MeshData& data : freeList) {
            auto remaining = _RemainingBytes(data);
            // TODO: Instead choose smallest buffer that will work rather than first available?
            if (remaining >= bytes) {
                return &data;
            }
        }
        return nullptr;
    }

    uint32_t GpuMeshAllocator::_AllocateData(const uint32_t size, const size_t byteMultiplier, const size_t maxBytes, 
                                             GpuBuffer& buffer, _MeshData& data, std::vector<GpuMeshAllocator::_MeshData>& freeList) {
        assert(size > 0);

        _MeshData * dataPtr = &data;
        const size_t totalSizeBytes = size_t(size) * byteMultiplier;
        const size_t remainingBytes = _RemainingBytes(data);

        if (totalSizeBytes > remainingBytes) {
            // See if one of the slots has data we can use
            _MeshData * freeSlot = _FindFreeSlot(freeList, totalSizeBytes);
            if (freeSlot) {
                dataPtr = freeSlot;
            }
            // If not perform a resize
            else {
                const size_t newSizeBytes = data.lastByte + std::max(size_t(size), minVertices) * byteMultiplier;
                if (newSizeBytes > maxBytes) {
                    throw std::runtime_error("Maximum GpuMesh bytes exceeded");
                }
                _Resize(buffer, data, newSizeBytes);
            }
        }

        const uint32_t offset = dataPtr->nextByte / byteMultiplier;
        dataPtr->nextByte += totalSizeBytes;

        // We pulled from a free list - delete if empty
        if (dataPtr != &data && _RemainingBytes(*dataPtr) == 0) {
            std::stable_partition(freeList.begin(), freeList.end(),
                [dataPtr](_MeshData& d) { return &d != dataPtr; }
            );
            freeList.pop_back();
        }

        return offset;
    }

    uint32_t GpuMeshAllocator::AllocateVertexData(const uint32_t numVertices) {
        return _AllocateData(numVertices, sizeof(GpuMeshData), maxVertexBytes, _vertices, _lastVertex, _freeVertices);
    }

    uint32_t GpuMeshAllocator::AllocateIndexData(const uint32_t numIndices) {
        return _AllocateData(numIndices, sizeof(uint32_t), maxIndexBytes, _indices, _lastIndex, _freeIndices);
    }

    void GpuMeshAllocator::_DeallocateData(_MeshData& last, std::vector<GpuMeshAllocator::_MeshData>& freeList, const size_t offsetBytes, const size_t lastByte) {
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
                size_t numElements = 1;
                for (size_t i = 0; i < freeList.size() - 1; ++i) {
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
                        [](const _MeshData& d) { return _RemainingBytes(d) != 0; }
                    );
                    auto removed = std::distance(it, freeList.end());
                    for (int i = 0; i < removed; ++i) {
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

    void GpuMeshAllocator::DeallocateVertexData(const uint32_t offset, const uint32_t numVertices) {
        const size_t offsetBytes = offset * sizeof(GpuMeshData);
        const size_t lastByte = offsetBytes + numVertices * sizeof(GpuMeshData);
        _DeallocateData(_lastVertex, _freeVertices, offsetBytes, lastByte);
    }

    void GpuMeshAllocator::DeallocateIndexData(const uint32_t offset, const uint32_t numIndices) {
        const size_t offsetBytes = offset * sizeof(uint32_t);
        const size_t lastByte = offsetBytes + numIndices * sizeof(uint32_t);
        _DeallocateData(_lastIndex, _freeIndices, offsetBytes, lastByte);
    }

    void GpuMeshAllocator::CopyVertexData(const std::vector<GpuMeshData>& data, const uint32_t offset) {
        const intptr_t byteOffset = intptr_t(offset) * sizeof(GpuMeshData);
        _vertices.CopyDataToBuffer(byteOffset, data.size() * sizeof(GpuMeshData), (const void *)data.data());
    }

    void GpuMeshAllocator::CopyIndexData(const std::vector<uint32_t>& data, const uint32_t offset) {
        const intptr_t byteOffset = intptr_t(offset) * sizeof(uint32_t);
        _indices.CopyDataToBuffer(byteOffset, data.size() * sizeof(uint32_t), (const void *)data.data());
    }

    void GpuMeshAllocator::BindBase(const GpuBaseBindingPoint& point, const uint32_t index) {
        _vertices.BindBase(point, index);
    }

    void GpuMeshAllocator::BindElementArrayBuffer() {
        _indices.Bind(GpuBindingPoint::ELEMENT_ARRAY_BUFFER);
    }

    void GpuMeshAllocator::UnbindElementArrayBuffer() {
        _indices.Unbind(GpuBindingPoint::ELEMENT_ARRAY_BUFFER);
    }

    void GpuMeshAllocator::_Initialize() {
        if (_initialized) return;
        _initialized = true;
        _lastVertex.nextByte = 0;
        _lastIndex.nextByte = 0;
        _Resize(_vertices, _lastVertex, minVertices * sizeof(GpuMeshData));
        _Resize(_indices, _lastIndex, minVertices * sizeof(uint32_t));
    }

    void GpuMeshAllocator::_Shutdown() {
        _vertices = GpuBuffer();
        _indices = GpuBuffer();
        _initialized = false;
    }

    void GpuMeshAllocator::_Resize(GpuBuffer& buffer, _MeshData& data, const size_t newSizeBytes) {
        GpuBuffer resized = GpuBuffer(nullptr, newSizeBytes, GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE);
        // Null check
        if (buffer != GpuBuffer()) {
            resized.CopyDataFromBuffer(buffer);
        }
        data.lastByte = newSizeBytes;
        buffer = resized;
    }

    size_t GpuMeshAllocator::_RemainingBytes(const _MeshData& data) {
        return data.lastByte - data.nextByte;
    }

    uint32_t GpuMeshAllocator::FreeVertices() {
        uint32_t vertices = static_cast<uint32_t>(_RemainingBytes(_lastVertex) / sizeof(GpuMeshData));
        for (auto& data : _freeVertices) {
            vertices += static_cast<uint32_t>(_RemainingBytes(data) / sizeof(GpuMeshData));
        }
        return vertices;
    }

    uint32_t GpuMeshAllocator::FreeIndices() {
        uint32_t indices = static_cast<uint32_t>(_RemainingBytes(_lastIndex) / sizeof(uint32_t));
        for (auto& data : _freeIndices) {
            indices += static_cast<uint32_t>(_RemainingBytes(data) / sizeof(uint32_t));
        }
        return indices;
    }

    GpuCommandBuffer::GpuCommandBuffer() {

    }

    void GpuCommandBuffer::RemoveCommandsAt(const std::unordered_set<size_t>& indices) {
        _VerifyArraySizes();
        if (indices.size() == 0) return;

        // std::vector<uint64_t> newHandles;
        std::vector<uint32_t> newMaterialIndices;
        std::vector<glm::mat4> newModelTransforms;
        std::vector<GpuDrawElementsIndirectCommand> newIndirectDrawCommands;
        for (size_t i = 0; i < NumDrawCommands(); ++i) {
            if (indices.find(i) == indices.end()) {
                // handlesToIndicesMap.insert(std::make_pair(handles[i], newHandles.size()));
                // newHandles.push_back(handles[i]);
                newMaterialIndices.push_back(materialIndices[i]);
                newModelTransforms.push_back(modelTransforms[i]);
                newIndirectDrawCommands.push_back(indirectDrawCommands[i]);
            }
            // else {
            //     handlesToIndicesMap.erase(handles[i]);
            // }
        }

        // handles = std::move(newHandles);
        materialIndices = std::move(newMaterialIndices);
        modelTransforms = std::move(newModelTransforms);
        indirectDrawCommands = std::move(newIndirectDrawCommands);
    }

    size_t GpuCommandBuffer::NumDrawCommands() const {
        return indirectDrawCommands.size();
    }

    void GpuCommandBuffer::UploadDataToGpu() {
        _VerifyArraySizes();

        const size_t numElems = NumDrawCommands();
        if (numElems == 0) return;

        if (_indirectDrawCommands == GpuBuffer() || _indirectDrawCommands.SizeBytes() < (numElems * sizeof(GpuDrawElementsIndirectCommand))) {
            const Bitfield flags = GPU_DYNAMIC_DATA;

            _materialIndices = GpuBuffer((const void *)materialIndices.data(), numElems * sizeof(uint32_t), flags);
            _modelTransforms = GpuBuffer((const void *)modelTransforms.data(), numElems * sizeof(glm::mat4), flags);
            _indirectDrawCommands = GpuBuffer((const void *)indirectDrawCommands.data(), numElems * sizeof(GpuDrawElementsIndirectCommand), flags);
        }
        else {
            _materialIndices.CopyDataToBuffer(0, numElems * sizeof(uint32_t), (const void *)materialIndices.data());
            _modelTransforms.CopyDataToBuffer(0, numElems * sizeof(glm::mat4), (const void *)modelTransforms.data());
            _indirectDrawCommands.CopyDataToBuffer(0, numElems * sizeof(GpuDrawElementsIndirectCommand), (const void *)indirectDrawCommands.data());
        }
    }

    void GpuCommandBuffer::BindMaterialIndicesBuffer(uint32_t index) {
        if (_materialIndices == GpuBuffer()) {
            throw std::runtime_error("Null material indices GpuBuffer");
        }
        _materialIndices.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer::BindModelTransformBuffer(uint32_t index) {
        if (_modelTransforms == GpuBuffer()) {
            throw std::runtime_error("Null model transform GpuBuffer");
        }
        _modelTransforms.BindBase(GpuBaseBindingPoint::SHADER_STORAGE_BUFFER, index);
    }

    void GpuCommandBuffer::BindIndirectDrawCommands() {
        if (_indirectDrawCommands == GpuBuffer()) {
            throw std::runtime_error("Null indirect draw command buffer");
        }
        _indirectDrawCommands.Bind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);
    }

    void GpuCommandBuffer::UnbindIndirectDrawCommands() {
        if (_indirectDrawCommands == GpuBuffer()) {
            throw std::runtime_error("Null indirect draw command buffer");
        }
        _indirectDrawCommands.Unbind(GpuBindingPoint::DRAW_INDIRECT_BUFFER);
    }

    void GpuCommandBuffer::_VerifyArraySizes() const {
        assert(//materialIndices.size() == handlesToIndicesMap.size() &&
               //materialIndices.size() == handles.size() &&
               materialIndices.size() == modelTransforms.size() &&
               materialIndices.size() == indirectDrawCommands.size());

        assert(NumDrawCommands() <= _maxDrawCalls);
    }
}