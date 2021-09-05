#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>

namespace stratus {
    /**
     * A ConcurrentHashMap is a thread-safe hash table that attempts
     * to maximize the number of threads who are capable of modifying
     * the map at the same time. To accomplish this, a hint can be given to
     * the hash map to indicate the number of threads that will be accessing
     * the map. By default, this number is set to 16. If different threads
     * are accessing different segments of the hash table, they will not block
     * each other.
     *
     * This is not a lock-free solution. Locks are used to synchronize access
     * to internal hash table segments, but since there are multiple distinct segments
     * this has the chance to significantly reduce the amount of contention
     * that is present. As a result, threads can spend less time waiting for locks
     * to become available.
     *
     * Guarantees made by this map:
     *      1) So long as the hash map has not gone out of scope, no iterator will
     *         ever be invalidated.
     *      2) Iterators themselves are weakly consistent. Many changes can
     *         happen to the map from the time an iterator is first created to the
     *         time it reaches the end. So, the only thing you can be sure of is that
     *         an iterator will not show you the same element multiple times, and an
     *         iterator will never become invalid unless the map goes out of scope.
     *      3) Contains(key) is also weakly consistent and represents the state of the
     *         map at the time the method was called. "if (Map.Contains(key)) Map.Insert(key, value)"
     *         is not guaranteed to be correct as some other thread could have inserted
     *         that key into the map between the two method calls.
     *      4) Size() is also weakly consistent in the same way as Contains(key). The size
     *         of the map could be different immediately after Size() returns.
     *      5) Insert(key, val) will override existing values. If the key already exists, it will
     *         be overridden. If multiple threads are inserting an existing key at
     *         the same time, the last thread to acquire the lock for the segment that
     *         the key belongs in will be the resulting state of that entry.
     *      6) InsertIfAbsent(key, val) provides the strong guarantee that it will never override
     *         an existing element. It is the safe version of
     *         "if (Map.Contains(key)) Map.Insert(key, value)".
     *      7) For a variety of reasons largely related to performance, the number
     *         of internal segments are set in stone after the constructor is finished.
     *         Both copy/move operator= have been marked deleted.
     *      8) You can simultaneously check if a key exists as well as obtain its full key-value
     *         entry, you can do something like "auto entry = Map.Find(key);
     *         if (entry != Map.end()) *use entry*;"
     *
     * @author Justin Hall
     */
    template<typename K,
        typename V,
        typename H = std::hash<K>>
    class ConcurrentHashMap {
    public:
        typedef K key_type;
        typedef V value_type;
        typedef size_t size_type;
        typedef std::pair<key_type, value_type> entry_type;
        typedef std::pair<const key_type, value_type> const_entry_type;

    private:
        struct Entry {
            /**
             * Contains the key-value pair, where entry.first is the key
             * and entry.second is the value. This pair is considered to be
             * immutable to the map, meaning entry.first/second = x will never
             * take place. If the map needs to overwrite an existing entry, it
             * will construct a whole new Entry object with its own entry shared
             * pointer.
             */
            std::shared_ptr<const_entry_type> entry;

            /**
             * Cached hash code so that we don't ever need to recalculate it
             */
            const size_t hashCode;

            Entry(const std::shared_ptr<const_entry_type> & entry,
                size_t hashCode)
                :
                entry(entry),
                hashCode(hashCode) { }

            Entry(std::shared_ptr<const_entry_type> && entry,
                size_t hashCode)
                :
                entry(std::forward<std::shared_ptr<const_entry_type>>(entry)),
                hashCode(hashCode) { }

            Entry(const entry_type & entry,
                size_t hashCode)
                :
                entry(std::make_shared<const_entry_type>(entry)),
                hashCode(hashCode) { }

            Entry(entry_type && entry,
                size_t hashCode)
                :
                entry(std::make_shared<const_entry_type>(std::forward<entry_type &&>(entry))),
                hashCode(hashCode) { }

            Entry(Entry && other) = delete;
            Entry(const Entry & other) = delete;

            const key_type & GetKey() const {
                return entry->first;
            }

            const value_type & GetValue() const {
                return entry->second;
            }
        };

        class ConcurrentIterator;
        struct Bucket {
            //typedef std::atomic<uint32_t> control_block_type;
            typedef uint8_t control_block_type;

            friend class ConcurrentIterator;

            /**
             * This represents a slot whose memory can be freely repurposed.
             */
            static const uint8_t EMPTY_SLOT = 0;

            /**
             * A busy slot could be empty, tombstone, or occupied. However, when a slot
             * is marked busy write, it is unsafe for readers to attempt to read the memory
             * or for other writers to secure a lock on the slot.
             */
            static const uint8_t BUSY_WRITE_SLOT = 1;

            /**
             * When a slot is marked busy read, other readers can safely secure the slot
             * and read also. However, 
             */
            static const uint8_t BUSY_READ_SLOT = 2;
            
            /**
             * A tombstone slot is a slot that was previously occupied but was then
             * deleted. Its memory can now be repurposed.
             */
            static const uint8_t TOMBSTONE_SLOT = 3;

            /**
             * This shows up during table resizes. Old entries will be marked as moved
             * to let functions like Contains() know that it would be a good idea
             * to retry their search.
             */
            static const uint8_t MOVED_SLOT = 4;

            /**
             * An occupied slot is one whose memory is used and is safe to access.
             */
            static const uint8_t OCCUPIED_SLOT = 5;

            /**
             * Since a control block is embedded within the buffer, to get to the next
             * actual entry you need to add ELEM_OFFSET to the current pointer location.
             */
            static constexpr size_type ELEM_OFFSET = sizeof(Entry) + sizeof(control_block_type);// +sizeof(size_type);

            /**
             * LOAD_FACTOR is used to compute the threshold, whhich is (capacity * LOAD_FACTOR). 
             * When the number of existing elements passes threshold, the table is resized.
             */
            static inline const double LOAD_FACTOR = 0.75;

            /**
             * Garbage-collected pointer to the table, represented
             * as a uint8_t buffer. This buffer contains extra embedded
             * data for the algorithm to use, and so different parts of the buffer
             * get interpreted as different types.
             */
            std::shared_ptr<uint8_t> table;

            /**
             * Represents how many elements can be stored if all slots are
             * filled. This needs to _always_ be a power of 2 or the indexing function
             * will stop working properly.
             */
            size_type capacity;

            /**
             * How many entries are actually present in the table.
             */
            size_type numEntries = 0;

            /**
             * Since capacity is a power of two, mask is set to (capacity - 1) so that
             * we can do (& mask) instead of (% capacity) to determine the index for
             * a given hash code.
             */
            size_type mask;

            /**
             * Used to determine when to resize the table. It is set to (capacity * LOAD_FACTOR),
             * and when the number of elements passes this threshold the table is resized.
             */
            size_type threshold;

            /**
             * Represents how many times we can linearly probe the table before we consider
             * it a miss, meaning either the element doesn't exist or we need to resize the table.
             */
            size_type reprobeLimit;

            /**
             * This is used to temporarily lock the buckets to prevent anyone from
             * destroying the raw buffer that it maintains.
             */
            mutable std::shared_mutex lock;

        public:
            Bucket(size_type capacity) {
                _InitMemory(capacity);
            }

            Bucket(Bucket && other) {
                operator=(std::forward<Bucket &&>(other));
            }

            Bucket & operator=(Bucket && other) {
                if (this == &other) return *this;
                for (;;) {
                    std::unique_lock<std::shared_mutex> ulOther(other.lock);
                    std::unique_lock<std::shared_mutex> ulThis(lock, std::defer_lock);
                    if (!ulThis.try_lock()) continue; // Try again to prevent deadlock
                    _Move(std::forward<Bucket &&>(other));
                    other._InitMemory(16); // Reset the other Bucket to a default state of 16 empty slots
                    break;
                }
                return *this;
            }

            size_type Size() const {
                //std::shared_lock<std::shared_mutex> sl(lock);
                return numEntries;
            }

            bool Contains(const key_type & key, size_t hashCode) const {
                std::shared_lock<std::shared_mutex> sl(lock);
                hashCode = _Hash(hashCode);
                auto[entry, index, oldVal] = _FindSlotForRead(key, hashCode);
                return oldVal == OCCUPIED_SLOT;
            }

            std::pair<std::shared_ptr<const_entry_type>, size_type> Find(const key_type & key,
                                                        size_t hashCode) const {
                std::shared_lock<std::shared_mutex> sl(lock);
                hashCode = _Hash(hashCode);
                auto[entry, index, marker] = _FindSlotForRead(key, hashCode);
                return (marker == OCCUPIED_SLOT) ? std::make_pair(entry->entry, index) :
                    std::make_pair(nullptr, 0);
            }

            template<typename E>
            bool Insert(E && entry, size_t hashCode) {
                std::unique_lock<std::shared_mutex> ul(lock);
                if (Size() > threshold) _Rehash(capacity << 1);
                hashCode = _Hash(hashCode);
                return std::get<2>(_Insert<E &&>(std::forward<E &&>(entry), hashCode));
            }

            template<typename E>
            bool InsertIfAbsent(E && entry, size_t hashCode) {
                std::unique_lock<std::shared_mutex> ul(lock);
                if (Size() > threshold) _Rehash(capacity << 1);
                hashCode = _Hash(hashCode);
                return std::get<2>(_InsertIfAbsent<E &&>(std::forward<E &&>(entry), hashCode));
            }

            std::shared_ptr<const_entry_type> Remove(const key_type & key, size_t hashCode) {
                std::unique_lock<std::shared_mutex> ul(lock);
                hashCode = _Hash(hashCode);
                auto[entry, index, oldVal] = _FindSlotForRead(key, hashCode);
                if (entry == nullptr) return std::shared_ptr<const_entry_type>();
                if (oldVal != OCCUPIED_SLOT) {
                    _ReleaseEntry(entry, oldVal);
                    return std::shared_ptr<const_entry_type>();
                }
                else {
                    //entry->~Entry();
                    std::shared_ptr<const_entry_type> e = entry->entry;
                    _ReleaseEntry(entry, TOMBSTONE_SLOT);
                    --numEntries;
                    auto tab = table.get();
                    if (*_GetControlBlock(_GetEntry(tab, _NextIndex(index, 1))) ==
                        EMPTY_SLOT) {
                        _ReclaimTombstones(entry, index);
                    }
                    return e;
                }
            }

            void Clear() {
                std::unique_lock<std::shared_mutex> ul(lock);
                uint8_t * tab = table.get();
                for (size_type i = 0; i < capacity; ++i) {
                    auto entry = _GetEntry(tab, i);
                    auto control = _GetControlBlock(entry);
                    auto markerVal = *control;
                    if (markerVal == OCCUPIED_SLOT || markerVal == TOMBSTONE_SLOT) entry->~Entry();
                    _ReleaseEntry(entry, EMPTY_SLOT);
                }
                numEntries = 0;
            }

        private:
            void _ReclaimTombstones(Entry * start, size_type index) {
                //std::cout << "RECLAIMING!\n";
                auto tab = table.get();
                auto curr = start;
                auto control = _GetControlBlock(curr);
                while (*control == TOMBSTONE_SLOT) {
                    curr->~Entry();
                    *control = EMPTY_SLOT;
                    index = _PrevIndex(index, 1);
                    curr = _GetEntry(tab, index);
                    control = _GetControlBlock(curr);
                }
            }

            template<typename E>
            std::tuple<Entry *, size_type, bool> _Insert(E && entry, size_t hashCode) {
                auto[entryPtr, index, oldVal] = _FindSlotForEntry(_GetKey(entry), hashCode);
                bool existing = false;
                if (oldVal == OCCUPIED_SLOT) {
                    existing = true;
                    entryPtr->~Entry();
                }
                new ((void *)entryPtr) Entry(std::forward<E &&>(entry), hashCode);
                _ReleaseEntry(entryPtr, OCCUPIED_SLOT);
                if (!existing) {
                    auto cachedNumEntries = numEntries;
                    numEntries = cachedNumEntries + 1;
                }
                return std::make_tuple(entryPtr, index, !existing);
            }

            template<typename E>
            std::tuple<Entry *, size_type, bool> _InsertIfAbsent(E && entry, size_t hashCode) {
                auto[entryPtr, index, oldVal] = _FindSlotForEntry(_GetKey(entry), hashCode);
                if (oldVal == OCCUPIED_SLOT) return std::make_tuple(entryPtr, index, false);
                new ((void *)entryPtr) Entry(std::forward<E &&>(entry), hashCode);
                _ReleaseEntry(entryPtr, OCCUPIED_SLOT);
                auto cachedNumEntries = numEntries;
                numEntries = cachedNumEntries + 1;
                return std::make_tuple(entryPtr, index, true);
            }

            std::tuple<Entry *, size_type, uint32_t> _FindSlotForEntry(const key_type & key, size_t hashCode) {
                auto result = _FindSlotForRead(key, hashCode);
                if (std::get<0>(result) != nullptr) return result;
                _Rehash(capacity << 1);
                return _FindSlotForEntry(key, hashCode);
            }

            std::tuple<Entry *, size_type, uint32_t> _FindSlotForRead(const key_type & key, size_t hashCode) const {
                uint8_t * tab = table.get();
                auto index = _GetIndex(hashCode, mask);
                auto startIndex = index;
                auto rootEntry = _GetEntry(tab, index);
                //auto collisions = _getCollisionCount(rootEntry);
                size_t reprobes = 0;
                for (auto entry = rootEntry; reprobes < reprobeLimit; ++reprobes,
                    index = _NextIndex(startIndex, reprobes),
                    entry = _GetEntry(tab, index)) {
                    auto control = _GetControlBlock(entry);
                    auto oldVal = *control;// control->load();
                    //auto[entry, oldVal] = _acquireEntry(tab, index, SlotAcquireType::WRITE);
                    if (oldVal == EMPTY_SLOT || (oldVal == OCCUPIED_SLOT && entry->hashCode == hashCode && entry->GetKey() == key)) {
                        // Don't release it!!
                        return std::make_tuple(entry, index, oldVal);
                    }
                }
                return std::make_tuple(nullptr, 0, 0);
            }

            static const key_type & _GetKey(const entry_type & entry) {
                return entry.first;
            }

            static const key_type & _GetKey(const std::shared_ptr<const_entry_type> & entry) {
                return entry->first;
            }

            static const value_type & _GetValue(const entry_type & entry) {
                return entry.second;
            }

            static const value_type & _GetValue(const std::shared_ptr<const_entry_type> & entry) {
                return entry->second;
            }

            static void _ReleaseEntry(Entry * entry, uint32_t newTag) {
                auto control = _GetControlBlock(entry);
                *control = newTag;
                //control->store(newTag);
            }
            
            static Entry * _GetEntry(uint8_t * table, size_type index) {
                return (Entry *)(table + ELEM_OFFSET * index);
            }

            static control_block_type * _GetControlBlock(uint8_t * table, size_type index) {
                return (control_block_type *)(table + ELEM_OFFSET * index + sizeof(Entry));
            }

            static control_block_type * _GetControlBlock(Entry * entry) {
                return (control_block_type *)(entry + 1);
            }

            size_t _Hash(size_t hashCode) const {
                static std::hash<size_t> hashFunc;
                //return hashFunc(hashCode);
                //return hashCode;
                static size_t halfSizeTBits = sizeof(size_t) * 8 / 2;
                static size_t upperBits = (~0) << halfSizeTBits;
                static size_t lowerBits = upperBits >> halfSizeTBits;
                return ((hashCode & upperBits) >> halfSizeTBits) + ((hashCode & lowerBits) << halfSizeTBits);
            }

            void _Rehash(size_type newCapacity) {
                //std::cout << "Resizing to " << newCapacity << std::endl;
                Bucket newBucket(newCapacity);
                uint8_t * tab = table.get();
                for (size_type i = 0; i < capacity; ++i) {
                    //auto[entry, oldVal] = _acquireEntry(tab, i, SlotAcquireType::WRITE);
                    auto entry = _GetEntry(tab, i);
                    auto control = _GetControlBlock(entry);
                    auto oldVal = *control;// control->load();
                    if (oldVal == OCCUPIED_SLOT) {
                        auto[newEntry, unused_, unused2_] = newBucket._Insert(std::move(entry->entry), entry->hashCode);
                        //entry->redirect = newEntry;
                    }
                    _ReleaseEntry(entry, MOVED_SLOT);
                }
                _Move(std::move(newBucket));
            }

            size_t _GetIndex(size_t hashCode, size_type mask) const {
                return hashCode & mask;
            }

            size_t _NextIndex(size_t index, size_t step) const {
                //index = index + step;
                //return index < capacity ? index : 0;
                return (index + step) & mask;
            }

            size_t _PrevIndex(size_t index, size_t step) const {
                //index = index + step;
                //return index < capacity ? index : 0;
                return (index - step) & mask;
            }

            void _Move(Bucket && other) {
                table = other.table;
                capacity = other.capacity;
                mask = other.mask;
                threshold = other.threshold;
                reprobeLimit = other.reprobeLimit;
                numEntries = other.numEntries;
                other.table = nullptr;
                other.capacity = 0;
                other.mask = 0;
                other.threshold = 0;
                other.reprobeLimit = 0;
                other.numEntries = 0;
            }

            void _InitMemory(size_type capacity) {
                this->capacity = capacity;
                this->mask = capacity - 1;
                this->reprobeLimit = capacity >> 2;
                this->threshold = capacity * LOAD_FACTOR;
                auto ptr = new uint8_t[sizeof(size_type) + ELEM_OFFSET * capacity]();
                size_type * internalCapacity = (size_type *)ptr;
                *internalCapacity = capacity;
                table = std::shared_ptr<uint8_t>(ptr + sizeof(size_type), _FreeMemory);
                ptr = table.get();
                for (size_type i = 0; i < capacity; ++i) {
                    auto control = _GetControlBlock(ptr, i);
                    //auto collisions = _getCollisionCount(ptr, i);
                    new (control) control_block_type(EMPTY_SLOT);
                    //new (collisions) size_t(0);
                }
            }

            static void _FreeMemory(uint8_t * ptr) {
                auto originalPtr = (ptr - sizeof(size_type));
                // First pull out the capacity information
                size_type capacity = *(size_type *)originalPtr;
                for (size_type i = 0; i < capacity; ++i, ptr += ELEM_OFFSET) {
                    auto entry = (Entry *)ptr;
                    auto control = (control_block_type *)(entry + 1);
                    auto markerVal = *control;
                    if (markerVal == OCCUPIED_SLOT || markerVal == TOMBSTONE_SLOT) entry->~Entry();
                    //if (control->load() == OCCUPIED_SLOT) entry->~Entry();
                    control->~control_block_type();
                }
                delete[] originalPtr;
            }
        };

        class ConcurrentIterator {
            /**
             * Const pointer to a concurrent hash map's list of buckets. This pointer
             * is not owned by this class and will not be modified/deleted by an iterator.
             */
            const std::vector<std::unique_ptr<Bucket>> * _buckets;

            /**
             * Iterator into the _buckets array. This is used to determine which
             * bucket we are currently on as well as to know when we have run
             * out of buckets to iterate over.
             */
            typename std::vector<std::unique_ptr<Bucket>>::const_iterator _iter;

            /**
             * Since each bucket maintains an internal segment which is implemented as a
             * closed hash table. _segmentIndex is used to know which location we are
             * into that hash table. Usually we set this value to be the end of the segment
             * when first initializing it and iterate backwards. This protects us from iterating
             * over the same element multiple times. The reason is that with power-of-two size
             * tables, a resize will cause elements to either stay in their same spot or move
             * up, but never down.
             */
            size_type _segmentIndex = 0;

            /**
             * Cached reference to the current table of the bucket we are working on iterating over.
             * Keeping this reference prevents it from being garbage collected while
             * we are still working with it.
             */
            std::shared_ptr<uint8_t> _currentTable;

            /**
             * Raw table pointer which is needed to call certain Bucket static functions. This is cached
             * just so that we don't need to make a function call every time we want to
             * get the pointer that _currentTable manages.
             */
            uint8_t * _currentTablePtr;

            /**
             * Current entry we are looking at from the current table. We need to store a strong
             * reference to it since the one inside the table itself isn't guaranteed to
             * stay valid once the table is unlocked.
             */
            std::shared_ptr<const_entry_type> _currentEntry;

        public:
            typedef std::input_iterator_tag iterator_category;
            typedef entry_type value_type;
            typedef size_t difference_type;
            typedef const_entry_type * pointer;
            typedef const_entry_type & reference;

            ConcurrentIterator(const std::vector<std::unique_ptr<Bucket>> & buckets,
                            typename std::vector<std::unique_ptr<Bucket>>::const_iterator iter,
                            size_type segmentIndex)
                :
                _buckets(&buckets),
                _iter(iter) {
                if (iter == buckets.end()) {
                    _segmentIndex = 0;
                    return;
                }
                _segmentIndex = segmentIndex;
                std::shared_lock<std::shared_mutex> sl((*iter)->lock);
                _GetTable();
                next(1);
            }

            ConcurrentIterator(const std::vector<std::unique_ptr<Bucket>> & buckets,
                            typename std::vector<std::unique_ptr<Bucket>>::const_iterator iter,
                            const std::shared_ptr<const_entry_type> & entry,
                            size_type segmentIndex)
                :
                _buckets(&buckets),
                _iter(iter),
                _segmentIndex(segmentIndex),
                _currentEntry(entry) {
                if (iter == buckets.end()) return;
                std::shared_lock<std::shared_mutex> sl((*iter)->lock);
                _GetTable();
            }

            pointer operator->() {
                return _currentEntry.get();
            }

            reference operator*() {
                return *_currentEntry;
            }

            const pointer operator->() const {
                return _currentEntry.get();
            }

            const reference operator*() const {
                return *_currentEntry;
            }

            ConcurrentIterator & operator++() {
                next(1);
                return *this;
            }

            ConcurrentIterator operator++(int) {
                ConcurrentIterator current(*this);
                next(1);
                return current;
            }

            ConcurrentIterator operator+(size_t index) const {
                ConcurrentIterator iter(*this);
                iter.next(index);
                return iter;
            }

            ConcurrentIterator & operator+=(size_t index) {
                next(index);
                return this->_iter;
            }

            bool operator==(const ConcurrentIterator & other) const {
                return _buckets == other._buckets &&
                    _iter == other._iter &&
                    _segmentIndex == other._segmentIndex;
            }

            bool operator!=(const ConcurrentIterator & other) const {
                return !(operator==(other));
            }

            void next(size_type distance) {
                if (_iter == _buckets->end()) return;
                std::shared_lock<std::shared_mutex> sl((*_iter)->lock);
                for (;;) {
                    if (_segmentIndex == 0) {
                        if ((_iter + 1) != _buckets->end()) {
                            ++_iter;
                            sl.unlock();
                            sl = std::shared_lock<std::shared_mutex>((*_iter)->lock);
                            _GetTable();
                            _segmentIndex = (*_iter)->capacity;
                        }
                        else {
                            _iter = _buckets->end();
                            _currentEntry = nullptr;
                            break;
                        }
                    }
                    --_segmentIndex;
                    auto entry = Bucket::_GetEntry(_currentTablePtr, _segmentIndex);
                    auto control = Bucket::_GetControlBlock(entry);
                    auto marker = *control;
                    if (marker == Bucket::MOVED_SLOT) {
                        ++_segmentIndex;
                        _GetTable();
                        continue;
                    }
                    if (marker == Bucket::OCCUPIED_SLOT) {
                        _currentEntry = entry->entry;
                        --distance;
                        if (distance == 0) break;
                    }
                }
            }

        private:
            void _GetTable() {
                _currentTable = (*_iter)->table;
                _currentTablePtr = _currentTable.get();
            }
        };
        
        /**
         * List of buckets. For performance reasons, this is stored as a normal
         * 1D array that should only ever be changed inside of the constructors.
         * Outside of them, deleting/adding buckets is completely unsafe. As a consequence,
         * copy/move operations are available at the constructor level, but both
         * copy/move operator= are marked deleted.
         */
        std::vector<std::unique_ptr<Bucket>> _buckets;

        /**
         * Hash function
         */
        H _hashFunc;

        /**
         * This essentially marks the theoretical number of threads that can modify
         * the hash map without blocking each other. This is achieved by creating one
         * bucket for each thread, and if different threads write to different buckets then
         * they will not block each other.
         */
        size_type _numThreads;

        /**
         * Concurrent hash map ensures that both the number of threads and the internal
         * bucket capacities are always a power of two. As a result, an optimization can be
         * made such that (hashCode % capacity) becomes (hashCode % (capacity - 1)) to determine
         * the index for an entry. The value of capacity - 1 is stored in _mask.
         */
        size_type _mask;

    public:
        typedef ConcurrentIterator iterator;
        typedef ConcurrentIterator const_iterator;

        ConcurrentHashMap(size_type capacity = 16, size_type numThreads = 16) {
            _numThreads = _RoundToPowerOf2(numThreads);
            capacity = _RoundToPowerOf2(capacity);
            capacity = (capacity < 16) ? 16 : capacity;
            _mask = _numThreads - 1;
            for (size_type i = 0; i < _numThreads; ++i) {
                _buckets.push_back(std::make_unique<Bucket>(capacity));
            }
        }

        ConcurrentHashMap(std::initializer_list<const_entry_type> ls)
            :
            ConcurrentHashMap(ls.Size()) {
            Insert(ls.begin(), ls.end());
        }

        ConcurrentHashMap(const ConcurrentHashMap & other) {
            _numThreads = other._numThreads;
            _mask = other._mask;
            for (size_type i = 0; i < _numThreads; ++i) {
                _buckets.push_back(std::make_unique<Bucket>(other._buckets[i]->capacity));
            }
            Insert(other.begin(), other.end());
        }

        ConcurrentHashMap(ConcurrentHashMap && other) {
            _numThreads = other._numThreads;
            _mask = other._mask;
            for (size_type i = 0; i < _numThreads; ++i) {
                _buckets.push_back(std::make_unique<Bucket>(std::move(*other._buckets[i])));
            }
        }

        ConcurrentHashMap & operator=(const ConcurrentHashMap & other) = delete;
        ConcurrentHashMap & operator=(ConcurrentHashMap && other) = delete;

        size_type Size() const {
            size_type numElems = 0;
            for (auto & bucket : _buckets) {
                numElems += bucket->Size();
            }
            return numElems;
        }

        bool Empty() const {
            return Size() == 0;
        }

        iterator Begin() {
            auto iter = _buckets.begin();
            return iterator(_buckets, iter, (*iter)->capacity);
        }

        iterator End() {
            return iterator(_buckets, _buckets.end(), nullptr, 0);
        }

        const_iterator Begin() const {
            auto iter = _buckets.begin();
            return const_iterator(_buckets, iter, (*iter)->capacity);
        }

        const_iterator End() const {
            return const_iterator(_buckets, _buckets.end(), nullptr, 0);
        }

        bool Contains(const key_type & key) const {
            auto hashCode = _Hash(key);
            auto index = _GetIndex(hashCode);
            return _buckets[index]->Contains(key, hashCode);
        }

        iterator Find(const key_type & key) {
            auto hashCode = _Hash(key);
            auto index = _GetIndex(hashCode);
            auto[entry, entryIndex] = _buckets[index]->Find(key, hashCode);
            if (entry == nullptr) return End();
            return iterator(_buckets, _buckets.begin() + index, entry, entryIndex);
        }

        const_iterator Find(const key_type & key) const {
            auto hashCode = _Hash(key);
            auto index = _GetIndex(hashCode);
            auto[entry, entryIndex] = _buckets[index]->Find(key, hashCode);
            if (entry == nullptr) return End();
            return const_iterator(_buckets, _buckets.begin() + index, entry, entryIndex);
        }

        bool Insert(const entry_type & entry) {
            auto hashCode = _Hash(entry.first);
            auto index = _GetIndex(hashCode);
            return _buckets[index]->Insert(entry, hashCode);
        }

        bool Insert(entry_type && entry) {
            auto hashCode = _Hash(entry.first);
            auto index = _GetIndex(hashCode);
            return _buckets[index]->Insert(std::forward<entry_type>(entry), hashCode);
        }

        template<typename Iterator>
        void Insert(Iterator first, Iterator last) {
            for (; first != last; ++first) {
                Insert(*first);
            }
        }

        bool InsertIfAbsent(const entry_type & entry) {
            auto hashCode = _Hash(entry.first);
            auto index = _GetIndex(hashCode);
            return _buckets[index]->InsertIfAbsent(entry, hashCode);
        }

        bool InsertIfAbsent(entry_type && entry) {
            auto hashCode = _Hash(entry.first);
            auto index = _GetIndex(hashCode);
            return _buckets[index]->InsertIfAbsent(std::forward<entry_type>(entry), hashCode);
        }

        template<typename Iterator>
        void InsertIfAbsent(Iterator first, Iterator last) {
            for (; first != last; ++first) {
                InsertIfAbsent(*first);
            }
        }

        std::shared_ptr<const_entry_type> Remove(const key_type & entry) {
            auto hashCode = _Hash(entry);
            auto index = _GetIndex(hashCode);
            return _buckets[index]->Remove(entry, hashCode);
        }

        void Clear() {
            for (auto & bucket : _buckets) {
                bucket->Clear();
            }
        }

        bool operator==(const ConcurrentHashMap & other) const {
            if (Size() != other.Size()) return false;
            for (auto&[key, value] : other) {
                if (!Contains(key)) return false;
            }
            return true;
        }

        bool operator!=(const ConcurrentHashMap & other) const {
            return !(operator==(other));
        }

    private:
        size_t _Hash(const key_type & key) const {
            return _hashFunc(key);
        }

        size_t _GetIndex(size_t hashCode) const {
            return hashCode & _mask;
        }

        // @see https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
        static size_type _RoundToPowerOf2(size_type val) {
            auto bits = sizeof(size_type) * 8;
            auto halfBits = bits / 2;
            --val;
            for (auto shift = 1; shift <= halfBits; shift <<= 1) {
                val |= val >> shift;
            }
            /**
            val |= val >> 1;
            val |= val >> 2;
            val |= val >> 4;
            val |= val >> 8;
            val |= val >> 16;
            val |= val >> 32;
            */
            val++;
            return val;
        }
    };
}