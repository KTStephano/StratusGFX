#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>

#include "StratusPoolAllocator.h"

TEST_CASE( "Stratus Pool Allocators Test", "[stratus_pool_allocators_test]" ) {
    std::cout << "StratusPoolAllocatorsTest" << std::endl;
    std::cout << stratus::ThreadUnsafePoolAllocator<int>::BytesPerElem << std::endl;
    auto pool = stratus::ThreadUnsafePoolAllocator<int>();

    int * ptr = pool.Malloc();
    *ptr = 25;
    std::cout << *ptr << std::endl;
    pool.Free(ptr);

    std::vector<int *> ptrs;
    constexpr int count = 1000000;
    for (int i = 0; i < count; ++i) {
        ptr = pool.Malloc();
        *ptr = i;
        ptrs.push_back(ptr);
    }

    for (int i = 0; i < count; ++i) {
        REQUIRE(*ptrs[i] == i);
    }

    std::cout << pool.NumChunks() << ", " << pool.NumElems() << std::endl;
}