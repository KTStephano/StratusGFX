#pragma once

#include "GL/gl3w.h"
#include "glm/glm.hpp"
#include <list>
#include <memory>
#include <vector>
#include <forward_list>
#include <cstdint>
#include "StratusCommon.h"
#include "StratusGpuCommon.h"
#include <unordered_set>
#include "StratusLog.h"
#include <list>
#include "StratusTypes.h"

#define MINIMUM_GPU_BLOCK_SIZE 64
// 2^30
#define MAX_GPU_BLOCK_SIZE 1073741824

namespace stratus {
    enum class GpuBindingPoint : i32 {
        // Good for things like vertices and normals
        ARRAY_BUFFER            = BITMASK64_POW2(1),
        // Good for indices
        ELEMENT_ARRAY_BUFFER    = BITMASK64_POW2(2),
        // Allows read-only uniform buffer access
        UNIFORM_BUFFER          = BITMASK64_POW2(3),
        // Allows read and write shader buffer access
        SHADER_STORAGE_BUFFER   = BITMASK64_POW2(4),
        // Allows for indirect array and element draw commands
        DRAW_INDIRECT_BUFFER    = BITMASK64_POW2(5)
    };

    // A more restrictive set of bindings good for things like floating point (vertex, normal, etc.)
    // and integer (index) buffers
    enum class GpuPrimitiveBindingPoint : i32 {
        ARRAY_BUFFER            = i32(GpuBindingPoint::ARRAY_BUFFER),
        ELEMENT_ARRAY_BUFFER    = i32(GpuBindingPoint::ELEMENT_ARRAY_BUFFER)
    };

    enum class GpuBaseBindingPoint : i32 {
        UNIFORM_BUFFER          = i32(GpuBindingPoint::UNIFORM_BUFFER),
        SHADER_STORAGE_BUFFER   = i32(GpuBindingPoint::SHADER_STORAGE_BUFFER)
    };

    enum class GpuStorageType : i32 {
        BYTE,
        UNSIGNED_BYTE,
        SHORT,
        UNSIGNED_SHORT,
        INT,
        UNSIGNED_INT,
        FLOAT
    };

    typedef i32 Bitfield;

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
    // Memory writes between client and server will be seen
    constexpr Bitfield GPU_MAP_COHERENT = BITMASK64_POW2(5);

    struct GpuBufferImpl;
    struct GpuArrayBufferImpl;

    // A gpu buffer holds primitive data usually in the form of floats, ints and shorts
    // TODO: Look into use cases for things other than STATIC_DRAW
    struct GpuBuffer {
        GpuBuffer() {}
        GpuBuffer(const void * data, const usize sizeBytes, const Bitfield usage = GPU_MAP_READ | GPU_MAP_WRITE);
        virtual ~GpuBuffer() = default;

        void EnableAttribute(i32 attribute, i32 sizePerElem, GpuStorageType, bool normalized, u32 stride, u32 offset, u32 divisor = 0);
        virtual void Bind(const GpuBindingPoint) const;
        virtual void Unbind(const GpuBindingPoint) const;
        // From what I can tell there shouldn't be a need to unbind UBOs
        virtual void BindBase(const GpuBaseBindingPoint, const u32 index) const;

        // Maps the GPU memory into system memory - make sure READ, WRITE, or PERSISTENT mapping is enabled
        void * MapMemory(const Bitfield access) const;
        void * MapMemory(const Bitfield access, isize offset, usize length) const;
        void UnmapMemory() const;
        bool IsMemoryMapped() const;

        usize SizeBytes() const;
        // Make sure GPU_DYNAMIC_DATA is set
        void CopyDataToBuffer(isize offset, usize size, const void * data);
        void CopyDataFromBuffer(const GpuBuffer&);
        void CopyDataFromBufferToSysMem(isize offset, usize size, void * data);

        // Memory mapping and data copying won't work after this
        void FinalizeMemory();

        bool operator==(const GpuBuffer& other) const {
            // Pointer comparison
            return this->impl_ == other.impl_;
        }

        bool operator!=(const GpuBuffer& other) const {
            return !(this->operator==(other));
        }

    protected:
        std::shared_ptr<GpuBufferImpl> impl_;
    };

    struct GpuPrimitiveBuffer final : public GpuBuffer {
        GpuPrimitiveBuffer() : GpuBuffer() {}
        GpuPrimitiveBuffer(const GpuPrimitiveBindingPoint type, const void * data, const usize sizeBytes, const Bitfield usage = 0);
        virtual ~GpuPrimitiveBuffer() = default;

        void Bind() const;
        void Unbind() const;

    private:
        GpuPrimitiveBindingPoint type_;
    };

    // Holds different gpu buffers and can bind/unbind them all as a group
    struct GpuArrayBuffer final {
        GpuArrayBuffer();
        ~GpuArrayBuffer() = default;

        void AddBuffer(const GpuPrimitiveBuffer&);
        usize GetNumBuffers() const;
        GpuPrimitiveBuffer& GetBuffer(usize);
        const GpuPrimitiveBuffer& GetBuffer(usize) const;
        void UnmapAllMemory() const;
        bool IsMemoryMapped() const;
        void FinalizeAllMemory();
        void Bind() const;
        void Unbind() const;
        void Clear();

    private:
        std::shared_ptr<std::vector<std::unique_ptr<GpuPrimitiveBuffer>>> buffers_;
    };

    // struct GpuTypedBufferMemoryPointer {
//     u32 index;
// };

    template<typename E>
    struct GpuTypedBuffer;

    template<typename E>
    using GpuTypedBufferPtr = std::shared_ptr<GpuTypedBuffer<E>>;

    // Manages a typed GPU memory pool. When an element is erased, that slot is marked for
    // reuse. The default object E() should be able to differentiate between used
    // and unused.
    template<typename E>
    struct GpuTypedBuffer {
        GpuTypedBuffer(usize blockSize, const bool allowResizing) 
            : allowResizing_(allowResizing) {
            blockSize_ = std::max<usize>(MINIMUM_GPU_BLOCK_SIZE, blockSize);
            //Resize_(blockSize_);
        }

        GpuTypedBuffer(GpuTypedBuffer&&) = default;
        GpuTypedBuffer(const GpuTypedBuffer&) = delete;

        GpuTypedBuffer& operator=(GpuTypedBuffer&&) = default;
        GpuTypedBuffer& operator=(const GpuTypedBuffer&) = delete;

        // Changes are buffered on the CPU
        void UploadChangesToGpu() {
            if (firstModifiedIndex_ != lastModifiedIndex_) {
                const isize offsetBytes = isize(firstModifiedIndex_) * sizeof(E);
                const usize sizeBytes = usize(lastModifiedIndex_ - firstModifiedIndex_) * sizeof(E);
                const void * data = (const void *)(cpuMemory_.data() + firstModifiedIndex_);
                gpuMemory_.CopyDataToBuffer(offsetBytes, sizeBytes, data);

                firstModifiedIndex_ = -1;
                lastModifiedIndex_ = -1;
            }
        }

        // Adds an element to either an existing slot
        // or to a new slot after resizing the buffer
        u32 Add(const E& elem) {
            if (NumFreeIndices() == 0) {
                Resize_(Capacity() + BlockSize());
            }

            auto next = freeIndices_.front();
            freeIndices_.pop_front();

            Set(elem, next);

            return next;
        }

        // Removes element at index (sets it to be equal to default E())
        void Remove(const u32 index) {
            Remove_(index, true);
        }

        // Marks entire memory region as free (sets everything to default E())
        void Clear() {
            for (u32 i = 0; i < capacity_; ++i) {
                Remove_(i, false);
            }

            maxIndex_ = 0;
        }

        // Pulls data from the CPU buffer for reading - may not
        // be in sync with the GPU memory if the GPU wrote to it
        const E& GetRead(const u32 index) const {
            if (index >= Capacity()) {
                throw std::runtime_error("Index exceeds capacity");
            }

            return cpuMemory_[index];
        }

        // Sets the element at index. If the index is beyond the bounds
        // of the current capacity it will attempt to resize it.
        void Set(const E& elem, const u32 index) {
            if (index >= Capacity()) {
                const usize offsetIndex = 1 + index - Capacity();
                usize multiplier = offsetIndex / BlockSize();
                multiplier += 1;

                Resize_(Capacity() + multiplier * BlockSize());
                Set(elem, index);
                return;
            }

            usedIndices_[index] = true;
            cpuMemory_[index] = elem;

            UpdateModifiedIndices_(static_cast<i32>(index));

            if (index >= maxIndex_) {
                maxIndex_ = index + 1;
            }
        }

        // Gets the underlying GpuBuffer for the entire memory region
        GpuBuffer GetBuffer() const {
            return gpuMemory_;
        }

        // Memory is allocated in fixed size blocks
        usize BlockSize() const {
            return blockSize_;
        }

        // This is an estimate of the current size. It returns the largest
        // index where data is occupied. The GPU will need to manually check
        // if each element before Size() - 1 is equal to E() or not.
        usize Size() const {
            return maxIndex_;
        }

        // Returns current capacity which may change if resizing is enabled
        usize Capacity() const {
            return capacity_;
        }

        // Returns how many memory slots are free for use
        usize NumFreeIndices() {
            return freeIndices_.size();
        }

        static inline GpuTypedBufferPtr<E> Create(const usize blockSize, const bool allowResizing) {
            return GpuTypedBufferPtr<E>(new GpuTypedBuffer<E>(blockSize, allowResizing));
        }

    private:
        void Resize_(const usize newSize) {
            if (newSize < capacity_) return;

            if (newSize > MAX_GPU_BLOCK_SIZE) {
                throw std::runtime_error("Max GPU block size exceeded");
            }

            if (capacity_ != 0 && !allowResizing_) {
                throw std::runtime_error("Ran out of free GPU memory (resizing was disabled)");
            }

            const Bitfield flags = GPU_DYNAMIC_DATA | GPU_MAP_READ | GPU_MAP_WRITE;
            cpuMemory_.resize(newSize, E());
            usedIndices_.resize(newSize, false);
            gpuMemory_ = GpuBuffer((const void*)cpuMemory_.data(), sizeof(E) * newSize, flags);

            for (usize i = capacity_; i < newSize; ++i) {
                freeIndices_.push_back(i);
            }

            capacity_ = newSize;
            // Reset these since we just copied everything over
            firstModifiedIndex_ = -1;
            lastModifiedIndex_ = -1;
        }

        void UpdateModifiedIndices_(const i32 index) {
            if (index <= firstModifiedIndex_ || firstModifiedIndex_ == -1) {
                firstModifiedIndex_ = index;
            }

            if (lastModifiedIndex_ <= index) {
                lastModifiedIndex_ = index + 1;
            }
        }

        void Remove_(const u32 index, const bool findNewMaxIndex) {
            if (index > capacity_ || usedIndices_[index] == false) return;

            if (NumFreeIndices() == 0 || index > freeIndices_.front()) {
                freeIndices_.push_back(index);
            }
            else {
                freeIndices_.push_front(index);
            }

            usedIndices_[index] = false;
            cpuMemory_[index] = E();

            UpdateModifiedIndices_(static_cast<i32>(index));

            if (findNewMaxIndex && (index + 1) == maxIndex_) {
                maxIndex_ = 0;
                for (i32 i = static_cast<i32>(index) - 1; i >= 0; --i) {
                    if (usedIndices_[i]) {
                        maxIndex_ = static_cast<usize>(i) + 1;
                        break;
                    }
                }
            }
        }

    private:
        std::vector<E> cpuMemory_;
        GpuBuffer gpuMemory_;
        std::list<u32> freeIndices_;
        std::vector<bool> usedIndices_;
        usize capacity_ = 0;
        usize blockSize_ = 0;
        usize maxIndex_ = 0;
        i32 firstModifiedIndex_ = -1;
        i32 lastModifiedIndex_ = -1;
        bool allowResizing_;
    };

    // Responsible for allocating vertex and index data. All data is stored
    // in two giant GPU buffers (one for vertices, one for indices).
    //
    // This is NOT thread safe as only the main thread should be using it since 
    // it performs GPU memory allocation.
    //
    // It can support a maximum of UINT_MAX vertices and UINT_MAX indices.
    class GpuMeshAllocator final {
        // This class initializes the global GPU memory for this class
        friend class GraphicsDriver;

        struct _MeshData {
            usize nextByte;
            usize lastByte;
        };

        GpuMeshAllocator() {}

    public:
        // Allocates 64-byte block vertex data where each element represents a GpuMeshData type.
        //
        // @return offset into global GPU vertex data array where data begins
        static u32 AllocateVertexData(const u32 numVertices);
        // @return offset into global GPU index data array where data begins
        static u32 AllocateIndexData(const u32 numIndices);

        // Deallocation
        static void DeallocateVertexData(const u32 offset, const u32 numVertices);
        static void DeallocateIndexData(const u32 offset, const u32 numIndices);

        static void CopyVertexData(const std::vector<GpuMeshData>&, const u32 offset);
        static void CopyIndexData(const std::vector<u32>&, const u32 offset);

        // Binds the GpuMesh buffer
        static void BindBase(const GpuBaseBindingPoint&, const u32);
        // Binds/unbinds indices buffer
        static void BindElementArrayBuffer();
        static void UnbindElementArrayBuffer();

        static u32 FreeVertices();
        static u32 FreeIndices();

    private:
        static _MeshData * FindFreeSlot_(std::vector<_MeshData>&, const usize bytes);
        static u32 AllocateData_(const u32 size, const usize byteMultiplier, const usize maxBytes, 
                                      GpuBuffer&, _MeshData&, std::vector<GpuMeshAllocator::_MeshData>&);
        static void DeallocateData_(_MeshData&, std::vector<GpuMeshAllocator::_MeshData>&, const usize offsetBytes, const usize lastByte);
        static void Initialize_();
        static void Shutdown_();
        static void Resize_(GpuBuffer& buffer, _MeshData& data, const usize newSizeBytes);
        static usize RemainingBytes_(const _MeshData& data);

    private:
        static GpuBuffer vertices_;
        static GpuBuffer indices_;
        // Allows for O(1) allocation when data is available
        static _MeshData lastVertex_;
        static _MeshData lastIndex_;
        // Allows for O(N) allocation by searching for previously deallocated
        // chunks of memory
        static std::vector<_MeshData> freeVertices_;
        static std::vector<_MeshData> freeIndices_;
        static bool initialized_;
    };
}
