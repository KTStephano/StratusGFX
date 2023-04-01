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
    template<typename E, typename Lock, size_t ElemsPerChunk, size_t Chunks>
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
                delete tmp;
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

    private:
        void InitChunk_() {
            Chunk_* c = new Chunk_();
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

    template<typename E, size_t ElemsPerChunk = 64, size_t Chunks = 1>
    struct PoolAllocator : public PoolAllocatorImpl_<E, NoOpLock_, ElemsPerChunk, Chunks> {
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

    template<typename E, size_t ElemsPerChunk = 64, size_t Chunks = 1>
    struct ThreadSafePoolAllocator {
        typedef PoolAllocatorImpl_<E, Lock_, ElemsPerChunk, Chunks> Allocator;
        static constexpr size_t BytesPerElem = Allocator::BytesPerElem;
        static constexpr size_t BytesPerChunk = Allocator::BytesPerChunk;

    private:
        // This is a control structure which allows us to keep track of which
        // thread pool allocators are still in use and which need to be deleted
        // (lightweight ref counted pointer)
        struct AllocatorData_ {
            std::atomic<size_t> counter;
            Allocator * allocator;

            AllocatorData_() {
                counter.store(0);
                allocator = new Allocator();
            }

            ~AllocatorData_() {
                delete allocator;
                allocator = nullptr;
            }
        };

    public:
        struct Deleter {
            AllocatorData_ ** allocator;

            Deleter(AllocatorData_ ** allocator)
                : allocator(allocator) { (*allocator)->counter.fetch_add(1); }

            Deleter(Deleter&& other)
                : allocator(other.allocator) { other.allocator = nullptr; }

            Deleter(const Deleter& other)
                : allocator(other.allocator) { (*allocator)->counter.fetch_add(1); }

            Deleter& operator=(Deleter&&) = delete;
            Deleter& operator=(const Deleter&) = delete;

            ~Deleter() {
                if (allocator) {
                    auto prev = (*allocator)->counter.fetch_sub(1);
                    if (prev <= 1) {
                        delete (*allocator);
                        *allocator = nullptr;
                    }
                }
            }

            void operator()(E * ptr) {
                (*allocator)->allocator->Deallocate(ptr);
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
            EnsureValid_();
            return UniquePtr(GetAllocator_()->Allocate(args...), Deleter(&_alloc));
        }

        template<typename ... Types>
        static SharedPtr AllocateShared(const Types&... args) {
            EnsureValid_();
            return SharedPtr(GetAllocator_()->Allocate(args...), Deleter(&_alloc));
        }

        template<typename Construct, typename ... Types>
        static UniquePtr AllocateCustomConstruct(Construct c, const Types&... args) {
            EnsureValid_();
            return UniquePtr(GetAllocator_()->AllocateCustomConstruct(c, args...), Deleter(&_alloc));
        }

        template<typename Construct, typename ... Types>
        static SharedPtr AllocateSharedCustomConstruct(Construct c, const Types&... args) {
            EnsureValid_();
            return SharedPtr(GetAllocator_()->AllocateCustomConstruct(c, args...), Deleter(&_alloc));
        }

        static size_t NumChunks() {
            if (!_alloc) return 0;
            return GetAllocator_()->NumChunks();
        }

        static size_t NumElems() {
            if (!_alloc) return 0;
            return GetAllocator_()->NumElems();
        }

        template<typename Base>
        static std::unique_ptr<Base, BaseDeleter<Base>> UniqueCast(UniquePtr& ptr) {
            auto deleter = ptr.get_deleter();
            E * orig = ptr.release();
            Base * base = dynamic_cast<E *>(orig);
            return std::unique_ptr<Base, BaseDeleter<Base>>(base, BaseDeleter<Base>(deleter));
        }

    private:
        static void EnsureValid_() {
            if (!_alloc) _alloc = new AllocatorData_();
        }

        static Allocator * GetAllocator_() {
            return _alloc->allocator;
        }

    private:
        inline thread_local static AllocatorData_ * _alloc = nullptr;
    };

    /* This implementation works but may be slightly less cache efficient due to
       embedding the control structure with the data itself.
    template<typename E, size_t ElemsPerChunk = 64, size_t Chunks = 1>
    struct ThreadSafePoolAllocator {
        // This is a control structure which allows us to keep track of which
        // thread pool allocators are still in use and which need to be deleted
        // (lightweight ref counted pointer)
        struct AllocatorData {
            std::atomic<size_t> counter = 0;
            void * allocator = nullptr;
        };

        // This stores the allocator data control structure alongside the element data
        // for a minimum of 16 bytes per object. 
        // Exact size per object is sizeof(void *) + min(8, sizeof(E)).
        struct ElementData {
            AllocatorData * allocator;
            E element;

            template<typename ... Types>
            ElementData(const Types&... args)
                : element(args...) {}

            ~ElementData() {
                element.~E();
            }
        };

        typedef __PoolAllocator<ElementData, Lock, ElemsPerChunk, Chunks> Allocator;
        static constexpr size_t BytesPerElem = Allocator::BytesPerElem;
        static constexpr size_t BytesPerChunk = Allocator::BytesPerChunk;

        struct Deleter {
            void operator()(E * ptr) {
                uint8_t * bytes = reinterpret_cast<uint8_t *>(ptr);
                // Back up to the start of ElementData so we can access the allocator data control structure
                ElementData * data = reinterpret_cast<ElementData *>(bytes - sizeof(AllocatorData *));
                AllocatorData * allocData = data->allocator;
                Allocator * allocator = _GetAllocator(allocData);
                allocator->Deallocate(data);
                // This will free the allocator if ref count has reached 0
                _DecrPoolRefCount(allocData);
            }
        };

        typedef std::unique_ptr<E, Deleter> UniquePtr;
        typedef std::shared_ptr<E> SharedPtr;

        ThreadSafePoolAllocator() {}

        template<typename ... Types>
        UniquePtr Allocate(const Types&... args) {
            ElementData * data = _Allocate(args...);
            return UniquePtr(&data->element);
        }

        template<typename ... Types>
        SharedPtr AllocateShared(const Types&... args) {
            ElementData * data = _Allocate(args...);
            return SharedPtr(&data->element, Deleter());
        }

        size_t NumChunks() {
            return _GetAllocator(_manager.allocator)->NumChunks();
        }

        size_t NumElems() {
            return _GetAllocator(_manager.allocator)->NumElems();
        }

    private:
        template<typename ... Types>
        ElementData * _Allocate(const Types&... args) {
            ElementData * data = _GetAllocator(_manager.allocator)->Allocate(args...);
            data->allocator = _manager.allocator;
            _IncrPoolRefCount(_manager.allocator);
            return data;
        }

        static void _IncrPoolRefCount(AllocatorData * a) {
            a->counter.fetch_add(1);
        }

        static void _IncrPoolRefCount(void * a) {
            _IncrPoolRefCount(reinterpret_cast<AllocatorData *>(a));
        }

        static void _DecrPoolRefCount(AllocatorData * a) {
            if (a == nullptr) return;
            auto prev = a->counter.fetch_sub(1);
            if (prev <= 1) {
                Allocator * ptr = _GetAllocator(a);
                delete ptr;
                delete a;
            }
        }

        static Allocator * _GetAllocator(AllocatorData * a) {
            return reinterpret_cast<Allocator *>(a->allocator);
        }

        struct _PoolManager {
            AllocatorData * allocator;

            _PoolManager(AllocatorData * allocator)
                : allocator(allocator) { 
                allocator->allocator = reinterpret_cast<void *>(new Allocator());
                _IncrPoolRefCount(allocator); 
            }

            _PoolManager(_PoolManager&& other)
                : allocator(other.allocator) { other.allocator = nullptr; }

            _PoolManager(const _PoolManager& other)
                : allocator(other.allocator) { _IncrPoolRefCount(allocator); }

            _PoolManager& operator=(_PoolManager&&) = delete;
            _PoolManager& operator=(const _PoolManager&) = delete;

            ~_PoolManager() {
                _DecrPoolRefCount(allocator);
            }
        };

        // This is the only root reference to the allocator which lives as long as the thread lives.
        // More references to the allocator are created with each allocation.
        inline thread_local static _PoolManager _manager = _PoolManager(new AllocatorData());
    };
    */
}