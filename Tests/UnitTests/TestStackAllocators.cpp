#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>
#include <chrono>
#include <vector>
#include <unordered_map>

#include "StratusStackAllocator.h"

TEST_CASE( "Stratus Stack Allocators Test", "[stratus_stack_allocators_test]" ) {
	std::cout << "Beginning stratus::StackAllocator tests" << std::endl;

	static constexpr size_t numInts = 2048;
	static constexpr size_t numBytes = sizeof(int) * numInts;

	auto allocator = stratus::MakeUnsafe<stratus::StackAllocator>(numBytes);

    REQUIRE(allocator->Capacity() == numBytes);
    REQUIRE(allocator->Capacity() == allocator->Remaining());

	for (int i = 0; i < numInts / 2; ++i) {
		auto mem = allocator->Allocate(sizeof(int));
		REQUIRE(allocator->Remaining() == (allocator->Capacity() - sizeof(int) * (i + 1)));
	}

	auto mem = allocator->Allocate(allocator->Remaining());
	REQUIRE(allocator->Remaining() == 0);

	bool overflowed = false;
	try {
		allocator->Allocate(1);
	}
	catch (const std::bad_alloc&) {
		overflowed = true;
	}
	REQUIRE(overflowed);

	REQUIRE(allocator->Remaining() == 0);
	allocator->Deallocate();
	REQUIRE(allocator->Remaining() == numBytes);

	auto poolAllocator = stratus::StackBasedPoolAllocator<int>(allocator);
	REQUIRE(poolAllocator.Capacity() == numInts);
	REQUIRE(poolAllocator.Remaining() == numInts);

	for (int i = 0; i < numInts; ++i) {
		poolAllocator.allocate(1);
		REQUIRE(poolAllocator.Remaining() == (numInts - (i + 1)));
	}

	REQUIRE(poolAllocator.Remaining() == 0);

	REQUIRE(poolAllocator.Allocator()->Remaining() == allocator->Remaining() / sizeof(int));
	poolAllocator.Allocator()->Deallocate();
	REQUIRE(poolAllocator.Remaining() == numInts);
	REQUIRE(allocator->Remaining() == numBytes);

	std::vector<int, stratus::StackBasedPoolAllocator<int>> vec(poolAllocator);
	REQUIRE(poolAllocator.Remaining() == (numInts - vec.capacity()));

	auto vec2 = vec;
	REQUIRE(poolAllocator.Remaining() == (numInts - (vec.capacity() + vec2.capacity())));
}