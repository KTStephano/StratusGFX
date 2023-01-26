#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>

#include "StratusPoolAllocator.h"

TEST_CASE( "Stratus Pool Allocators Test", "[stratus_pool_allocators_test]" ) {
    std::cout << "StratusPoolAllocatorsTest" << std::endl;
    std::cout << stratus::ThreadSafePoolAllocator<int>::BytesPerElem << std::endl;
    auto pool = stratus::ThreadSafePoolAllocator<int>();

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