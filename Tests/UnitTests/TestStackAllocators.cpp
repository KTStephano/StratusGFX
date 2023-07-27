#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>
#include <chrono>

#include "StratusStackAllocator.h"

TEST_CASE( "Stratus Stack Allocators Test", "[stratus_stack_allocators_test]" ) {
	std::cout << "Beginning stratus::StackAllocator tests" << std::endl;

	static constexpr size_t numInts = 100;
	static constexpr size_t numBytes = sizeof(int) * numInts;

	stratus::StackAllocator allocator(numBytes);

    REQUIRE(allocator.Capacity() == numBytes);
    REQUIRE(allocator.Capacity() == allocator.Remaining());

	for (int i = 0; i < numInts / 2; ++i) {
		auto mem = allocator.Allocate(sizeof(int));
		REQUIRE(allocator.Remaining() == (allocator.Capacity() - sizeof(int) * (i + 1)));
	}

	auto mem = allocator.Allocate(allocator.Remaining());
	REQUIRE(allocator.Remaining() == 0);

	bool overflowed = false;
	try {
		allocator.Allocate(1);
	}
	catch (const std::bad_alloc&) {
		overflowed = true;
	}
	REQUIRE(overflowed);

	REQUIRE(allocator.Remaining() == 0);
	allocator.Deallocate();
	REQUIRE(allocator.Remaining() == numBytes);
}