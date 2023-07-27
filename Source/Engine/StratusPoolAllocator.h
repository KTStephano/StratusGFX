#pragma once

#include <memory>
#include <algorithm>
#include <thread>
#include <shared_mutex>
#include <functional>
#include <atomic>

// See https://www.qt.io/blog/a-fast-and-thread-safe-pool-allocator-for-qt-part-1
// for some more information

namespace stratus {
    template<typename C>
    struct DefaultChunkAllocator_ final {
        static C * Allocate(const size_t count) {
            return (C *)std::malloc(sizeof(C) * count);
        }

        template<typename ... Args>
        static void Construct(C * at, Args&&... args) {
            uint8_t * memory = reinterpret_cast<uint8_t *>(at);
            ::new (memory) C(std::forward<Args>(args)...);
        }

        static void Deallocate(C * ptr, const size_t count) {
            std::free(reinterpret_cast<void *>(ptr));
        }

        static void Destroy(C * ptr) {
            ptr->~C();
        }
    };

    struct NoOpLock_ {
        struct NoOpLockHeld{};
        typedef NoOpLockHeld value_type;

        value_type LockRead() const {
            return value_type();
        }

        value_type LockWrite() const {
            return value_type();
        }
    };

    // Allocates memory of a pre-defined size provide optimal data locality
    // with zero fragmentation
    template<typename E, typename Lock, size_t ElemsPerChunk, size_t Chunks, template<typename C> typename ChunkAllocator>
    struct PoolAllocatorImpl_ {
        static_assert(ElemsPerChunk > 0);
        static_assert(Chunks > 0);

        // We need to at least be able to store the next pointer
        static constexpr size_t BytesPerElem = std::max<size_t>(sizeof(void *), sizeof(E));
        static constexpr size_t BytesPerChunk = BytesPerElem * ElemsPerChunk;

        PoolAllocatorImpl_() {
            for (size_t i = 0; i < Chunks; ++i) {
                InitChunk_();
            }
        }

        // Copying is not supported at all
        PoolAllocatorImpl_(PoolAllocatorImpl_&&) = delete;
        PoolAllocatorImpl_(const PoolAllocatorImpl_&) = delete;
        PoolAllocatorImpl_& operator=(PoolAllocatorImpl_&&) = delete;
        PoolAllocatorImpl_& operator=(const PoolAllocatorImpl_&) = delete;

        virtual ~PoolAllocatorImpl_() {
            Chunk_* c = chunks_;
            while (c != nullptr) {
                Chunk_* tmp = c;
                c = c->next;
                chunkAllocator_.Destroy(tmp);
                chunkAllocator_.Deallocate(tmp, 1);
            }

            frontBuffer_ = nullptr;
            chunks_ = nullptr;
        }

    private:
        template<typename ... Types>
        static E * PlacementNew_(uint8_t * memory, const Types&... args) {
            return new (memory) E(args...);
        }

    public:
        template<typename ... Types>
        E * Allocate(const Types&... args) {
            return AllocateCustomConstruct(PlacementNew_<Types...>, args...);
        }

        template<typename Construct, typename ... Types>
        E * AllocateCustomConstruct(Construct c, const Types&... args) {
            uint8_t * bytes = nullptr;
            {
                //auto wlf = _frontBufferLock.LockWrite();
                if (!frontBuffer_) {
                    auto wlb = backBufferLock_.LockWrite();
                    // If back buffer has free nodes, swap front with back
                    if (backBuffer_) {
                        frontBuffer_ = backBuffer_;
                        backBuffer_ = nullptr;
                    }
                    // If not allocate a new chunk of memory
                    else {
                        InitChunk_();
                    }
                }

                MemBlock_* next = frontBuffer_;
                frontBuffer_ = frontBuffer_->next;
                bytes = reinterpret_cast<uint8_t *>(next);
            }
            return c(bytes, args...);
        }

        void Deallocate(E * ptr) {
            if (ptr == nullptr) return;
            ptr->~E();
            auto wlb = backBufferLock_.LockWrite();
            uint8_t * bytes = reinterpret_cast<uint8_t *>(ptr);
            MemBlock_* b = reinterpret_cast<MemBlock_*>(bytes);
            b->next = backBuffer_;
            backBuffer_ = b;
        }

        size_t NumChunks() const {
            //auto sl = _frontBufferLock.LockRead();
            return numChunks_;
        }

        size_t NumElems() const {
            //auto sl = _frontBufferLock.LockRead();
            return numElems_;
        }

    private:
        // Each _MemBlock sits in front of BytesPerElem memory
        struct alignas(sizeof(void *)) MemBlock_ {
            MemBlock_ * next;
        };

        struct Chunk_ {
            alignas(E) uint8_t memory[BytesPerChunk];
            Chunk_ * next = nullptr;
        };

        //Lock _frontBufferLock;
        Lock backBufferLock_;
        // Having two allows us to largely decouple allocations from deallocations
        // and only synchronize the two when we run out of free memory
        //
        // (we are running with the assumption that only one thread can allocate but many
        //  can deallocate for the same allocator object)
        MemBlock_* frontBuffer_ = nullptr;
        MemBlock_* backBuffer_ = nullptr;
        Chunk_* chunks_ = nullptr;
        size_t numChunks_ = 0;
        size_t numElems_ = 0;
        ChunkAllocator<Chunk_> chunkAllocator_;

    private:
        void InitChunk_() {
            Chunk_* c = chunkAllocator_.Allocate(1);
            chunkAllocator_.Construct(c);
            ++numChunks_;
            numElems_ += ElemsPerChunk;

            // Start at the end and add to freelist in reverse order
            uint8_t * mem = c->memory + BytesPerElem * (ElemsPerChunk - 1);
            for (size_t i = ElemsPerChunk; i > 0; --i, mem -= BytesPerElem) {
                MemBlock_* b = reinterpret_cast<MemBlock_*>(mem);
                b->next = frontBuffer_;
                frontBuffer_ = b;
            }

            c->next = chunks_;
            chunks_ = c;
        }
    };

    template<typename E, size_t ElemsPerChunk = 64, size_t Chunks = 1, template<typename C> typename ChunkAllocator = DefaultChunkAllocator_>
    struct PoolAllocator : public PoolAllocatorImpl_<E, NoOpLock_, ElemsPerChunk, Chunks, ChunkAllocator> {
        virtual ~PoolAllocator() = default;
    };

    struct Lock_ {
        struct LockHeld {
            typedef void (*UnlockFunction)(std::shared_mutex *);
            UnlockFunction unlock;
            std::shared_mutex * m;

            LockHeld(UnlockFunction unlock,
                     std::shared_mutex * m)
                : unlock(unlock), m(m) {
            }

            ~LockHeld() {
                unlock(m);
            }
        };

        typedef LockHeld value_type;

        const std::thread::id owner;
        mutable std::shared_mutex m;

        Lock_(const std::thread::id& owner = std::this_thread::get_id())
            : owner(owner) {}

        value_type LockRead() const {
            LockShared_(&m);
            return value_type(UnlockShared_, &m);
        }
        
        value_type LockWrite() const {
            // Allowing the owner to always lock read gives a performance increase
            if (std::this_thread::get_id() == owner) {
                return LockRead();
            }
            LockUnique_(&m);
            return value_type(UnlockUnique_, &m);
        }

    private:
        static void LockShared_(std::shared_mutex * m) {
            m->lock_shared();
        }

        static void UnlockShared_(std::shared_mutex * m) {
            m->unlock_shared();
        }

        static void LockUnique_(std::shared_mutex * m) {
            m->lock();
        }

        static void UnlockUnique_(std::shared_mutex * m) {
            m->unlock();
        }
    };

    template<typename E, size_t ElemsPerChunk = 64, size_t Chunks = 1, template<typename C> typename ChunkAllocator = DefaultChunkAllocator_>
    struct ThreadSafePoolAllocator {
        typedef PoolAllocatorImpl_<E, Lock_, ElemsPerChunk, Chunks, ChunkAllocator> Allocator;
        static constexpr size_t BytesPerElem = Allocator::BytesPerElem;
        static constexpr size_t BytesPerChunk = Allocator::BytesPerChunk;

    public:
        struct Deleter {
            std::shared_ptr<Allocator> allocator;

            Deleter(const std::shared_ptr<Allocator>& allocator)
                : allocator(allocator) {}

            Deleter(Deleter&& other)
                : allocator(other.allocator) {}

            Deleter(const Deleter& other)
                : allocator(other.allocator) {}

            Deleter& operator=(Deleter&&) = delete;
            Deleter& operator=(const Deleter&) = delete;

            ~Deleter() {}

            void operator()(E * ptr) {
                //if (*allocator == nullptr) return;
                allocator->Deallocate(ptr);
            }
        };

        // If a pointer is cast to another this deleter is used
        template<typename Base>
        struct BaseDeleter {
            Deleter original;

            BaseDeleter(Deleter original)
                : original(original) {}

            void operator()(Base * ptr) {
                original(dynamic_cast<E *>(ptr));
            }
        };

        typedef std::unique_ptr<E, Deleter> UniquePtr;
        typedef std::shared_ptr<E> SharedPtr;

        ThreadSafePoolAllocator() {}

        template<typename ... Types>
        static UniquePtr Allocate(const Types&... args) {
            auto alloc = GetAllocator_();
            return UniquePtr(alloc->Allocate(args...), Deleter(alloc));
        }

        template<typename ... Types>
        static SharedPtr AllocateShared(const Types&... args) {
            auto alloc = GetAllocator_();
            return SharedPtr(alloc->Allocate(args...), Deleter(alloc));
        }

        template<typename Construct, typename ... Types>
        static UniquePtr AllocateCustomConstruct(Construct c, const Types&... args) {
            auto alloc = GetAllocator_();
            return UniquePtr(alloc->AllocateCustomConstruct(c, args...), Deleter(alloc));
        }

        template<typename Construct, typename ... Types>
        static SharedPtr AllocateSharedCustomConstruct(Construct c, const Types&... args) {
            auto alloc = GetAllocator_();
            return SharedPtr(alloc->AllocateCustomConstruct(c, args...), Deleter(alloc));
        }

        static size_t NumChunks() {
            auto alloc = WeakGetAllocator_();
            if (!alloc) return 0;
            return alloc->NumChunks();
        }

        static size_t NumElems() {
            auto alloc = WeakGetAllocator_();
            if (!alloc) return 0;
            return alloc->NumElems();
        }

        template<typename Base>
        static std::unique_ptr<Base, BaseDeleter<Base>> UniqueCast(UniquePtr& ptr) {
            auto deleter = ptr.get_deleter();
            E * orig = ptr.release();
            Base * base = dynamic_cast<E *>(orig);
            return std::unique_ptr<Base, BaseDeleter<Base>>(base, BaseDeleter<Base>(deleter));
        }

    private:
        static std::shared_ptr<Allocator> GetAllocator_() {
            auto alloc = alloc_.lock();
            if (!alloc) {
                alloc = std::make_shared<Allocator>();
                alloc_ = alloc;
            }
            return alloc;
        }

        static std::shared_ptr<Allocator> WeakGetAllocator_() {
            return alloc_.lock();
        }

    private:
        inline thread_local static std::weak_ptr<Allocator> alloc_;
    };
}