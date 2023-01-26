#pragma once

#include <memory>
#include <algorithm>

namespace stratus {
    // Allocates memory of a pre-defined size provide optimal data locality
    // with zero fragmentation.
    //
    // This is not designed for fast allocation speed, so try to keep
    // that to a minimum per frame.
    template<typename E, size_t ElemsPerChunk = 512, size_t Chunks = 1>
    struct ThreadUnsafePoolAllocator {
        static_assert(ElemsPerChunk > 0);
        static_assert(Chunks > 0);

        // We need to at least be able to store the next pointer
        static constexpr size_t BytesPerElem = std::max<size_t>(sizeof(void *), sizeof(E));
        static constexpr size_t BytesPerChunk = BytesPerElem * ElemsPerChunk;

        ThreadUnsafePoolAllocator() {
            for (size_t i = 0; i < Chunks; ++i) {
                _InitChunk();
            }
        }

        ~ThreadUnsafePoolAllocator() {
            _Chunk * c = _chunks;
            for (size_t i = 0; i < Chunks; ++i) {
                _Chunk * tmp = c;
                c = c->next;
                delete tmp;
            }

            _freeList = nullptr;
            _chunks = nullptr;
        }

        template<typename ... Types>
        E * Allocate(Types ... args) {
            if (!_freeList) {
                _InitChunk();
            }

            _MemBlock * next = _freeList;
            _freeList = _freeList->next;
            uint8_t * bytes = reinterpret_cast<uint8_t *>(next);
            return new (bytes) E(std::forward<Types>(args)...);
        }

        void Deallocate(E * ptr) {
            ptr->~E();
            uint8_t * bytes = reinterpret_cast<uint8_t *>(ptr);
            _MemBlock * b = reinterpret_cast<_MemBlock *>(bytes);
            b->next = _freeList;
            _freeList = b;
        }

        size_t NumChunks() const {
            return _numChunks;
        }

        size_t NumElems() const {
            return _numElems;
        }

    private:
        // Each _MemBlock sits in front of BytesPerElem memory
        struct alignas(sizeof(void *)) _MemBlock {
            _MemBlock * next;
        };

        struct _Chunk {
            uint8_t memory[BytesPerChunk];
            _Chunk * next = nullptr;
        };

        _MemBlock * _freeList = nullptr;
        _Chunk * _chunks = nullptr;
        size_t _numChunks = 0;
        size_t _numElems = 0;

    private:
        void _InitChunk() {
            _Chunk * c = new _Chunk();
            ++_numChunks;
            _numElems += ElemsPerChunk;

            // Start at the end and add to freelist in reverse order
            uint8_t * mem = c->memory + BytesPerElem * (ElemsPerChunk - 1);
            for (size_t i = ElemsPerChunk; i > 0; --i, mem -= BytesPerElem) {
                _MemBlock * b = reinterpret_cast<_MemBlock *>(mem);
                b->next = _freeList;
                _freeList = b;
            }

            c->next = _chunks;
            _chunks = c;
        }
    };
}