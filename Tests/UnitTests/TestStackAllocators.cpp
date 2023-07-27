#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>
#include <chrono>
#include <vector>
#include <unordered_map>

#include "StratusStackAllocator.h"

TEST_CASE( "Stratus Stack Allocators Test", "[stratus_stack_allocators_test]" ) {
	std::cout << "Beginning stratus::StackAllocator tests" << std::endl;

	// Should work out to 4 mb
	static constexpr size_t numInts = 1047552;
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

	{
		std::vector<int, stratus::StackBasedPoolAllocator<int>> vec(poolAllocator);
		REQUIRE(poolAllocator.Remaining() == (numInts - vec.capacity()));

		auto vec2 = vec;
		REQUIRE(poolAllocator.Remaining() == (numInts - (vec.capacity() + vec2.capacity())));
	}

	allocator->Deallocate();
	REQUIRE(poolAllocator.Remaining() == numInts);

	struct LargeUnevenStruct {
		// 117 bytes
		uint8_t array[117];
	};

	{
		auto map = std::unordered_map<
			double,
			double,
			std::hash<double>,
			std::equal_to<double>,
			stratus::StackBasedPoolAllocator<std::pair<const double, double>>>(
				8,
				stratus::StackBasedPoolAllocator<std::pair<const double, double>>(allocator)
			);

		poolAllocator = stratus::StackBasedPoolAllocator<int>(allocator);
		auto vec = std::vector<int, stratus::StackBasedPoolAllocator<int>>(poolAllocator);

		auto vec2 = std::vector<LargeUnevenStruct, stratus::StackBasedPoolAllocator<LargeUnevenStruct>>(
			stratus::StackBasedPoolAllocator<LargeUnevenStruct>(allocator)
		);

		REQUIRE((allocator->Remaining() > 0 && allocator->Remaining() < numBytes));

		for (int i = 0; i < 256; ++i) {
			map.insert(std::make_pair(double(i), double(i + 1)));
			vec.push_back(i);
			vec2.push_back(LargeUnevenStruct());
		}

		std::cout << allocator->Remaining() << std::endl;
	}
}