#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>

#include "StratusPoolAllocator.h"

static void ThreadUnsafePoolAllocatorTest() {
    std::cout << "ThreadUnsafePoolAllocatorTest" << std::endl;
    std::cout << stratus::ThreadUnsafePoolAllocator<int>::BytesPerElem << std::endl;
    auto pool = stratus::ThreadUnsafePoolAllocator<int>();

    int * ptr = pool.Allocate(25);
    REQUIRE(*ptr == 25);
    std::cout << *ptr << std::endl;
    pool.Deallocate(ptr);

    std::vector<int *> ptrs;
    constexpr int count = 1000000;
    for (int i = 0; i < count; ++i) {
        ptr = pool.Allocate(i);
        ptrs.push_back(ptr);
    }

    for (int i = 0; i < count; ++i) {
        REQUIRE(*ptrs[i] == i);
    }

    std::cout << pool.NumChunks() << ", " << pool.NumElems() << std::endl;
}

static void ThreadSafePoolAllocatorTest() {
    std::cout << "ThreadSafePoolAllocatorTest" << std::endl;
    typedef stratus::ThreadSafePoolAllocator<int> Allocator;
    std::cout << Allocator::BytesPerElem << std::endl;
    auto pool = Allocator();

    Allocator::SharedPtr ptr = pool.AllocateShared(25);
    REQUIRE(*ptr == 25);
    std::cout << *ptr << std::endl;

    std::vector<Allocator::SharedPtr> ptrs;
    constexpr int count = 1000000;
    for (int i = 0; i < count; ++i) {
        ptr = pool.AllocateShared(i);
        ptrs.push_back(ptr);
    }

    for (int i = 0; i < count; ++i) {
        REQUIRE(*ptrs[i] == i);
    }

    std::cout << pool.NumChunks() << ", " << pool.NumElems() << std::endl;
}

TEST_CASE( "Stratus Pool Allocators Test", "[stratus_pool_allocators_test]" ) {
    ThreadUnsafePoolAllocatorTest();
    ThreadSafePoolAllocatorTest();
}