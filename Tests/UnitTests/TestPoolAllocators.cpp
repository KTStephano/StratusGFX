#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>
#include <chrono>

#include "StratusPoolAllocator.h"

static void ThreadUnsafePoolAllocatorTest() {
    std::cout << "ThreadUnsafePoolAllocatorTest" << std::endl;
    std::cout << stratus::ThreadUnsafePoolAllocator<int64_t>::BytesPerElem << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto pool = stratus::ThreadUnsafePoolAllocator<int64_t>();

    int64_t * ptr = pool.Allocate(25);
    REQUIRE(*ptr == 25);
    std::cout << *ptr << std::endl;
    pool.Deallocate(ptr);

    std::vector<int64_t *> ptrs;
    constexpr int count = 16000000;
    for (int i = 0; i < count; ++i) {
        ptr = pool.Allocate(i);
        ptrs.push_back(ptr);
    }

    for (int i = 0; i < count; ++i) {
        REQUIRE(*ptrs[i] == i);
        pool.Deallocate(ptrs[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed MS: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    std::cout << pool.NumChunks() << ", " << pool.NumElems() << std::endl;
}

static void ThreadSafePoolAllocatorTest() {
    std::cout << "ThreadSafePoolAllocatorTest" << std::endl;
    typedef stratus::ThreadSafePoolAllocator<int64_t> Allocator;
    std::cout << Allocator::BytesPerElem << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto pool = Allocator();

    Allocator::UniquePtr ptr = pool.Allocate(int64_t(25));
    REQUIRE(*ptr == 25);
    std::cout << *ptr << std::endl;

    constexpr int64_t numThreads = 8;
    constexpr int64_t count = 16000000 / numThreads;
    std::vector<std::thread> threads;

    for (int64_t th = 0; th < numThreads; ++th) {
        threads.push_back(std::thread([&pool, &count]() {
            std::vector<Allocator::UniquePtr> ptrs;
            for (int64_t i = 0; i < count; ++i) {
                Allocator::UniquePtr ptr = pool.Allocate(i);
                ptrs.push_back(std::move(ptr));
            }

            for (int i = 0; i < count; ++i) {
                REQUIRE(*ptrs[i] == i);
                //ptrs[i].reset();
            }

            std::cout << pool.NumChunks() << ", " << pool.NumElems() << std::endl;
        }));
    }

    for (auto& th : threads) th.join();

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed MS: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
}

TEST_CASE( "Stratus Pool Allocators Test", "[stratus_pool_allocators_test]" ) {
    ThreadUnsafePoolAllocatorTest();
    ThreadSafePoolAllocatorTest();
}