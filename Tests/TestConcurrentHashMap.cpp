#include <catch2/catch_all.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#include "StratusConcurrentHashMap.h"

TEST_CASE( "Single threaded hash map", "[single_thread_hmap]" ) {
	std::cout << "Starting concurrent hash map test with a single thread" << std::endl;
	
	stratus::ConcurrentHashMap<int, int> map;

	REQUIRE(map.Size() == 0);
	REQUIRE(map.Empty() == true);

	const int numElements = std::thread::hardware_concurrency() * 10000;

	// Start the timer
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < numElements; ++i) {
		map.Insert(std::make_pair(i, i + 1));
		REQUIRE(map.Contains(i) == true);
		REQUIRE(map.Find(i)->second == (i + 1));
	}
	auto end = std::chrono::system_clock::now();

	REQUIRE(map.Size() == numElements);

	std::cout << "Single threaded test completed in " << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " msec" << std::endl;
	std::cout << "Comppleted a total of at least " << numElements << " read/write operations" << std::endl << std::endl;
}

TEST_CASE( "Multi threaded hash map", "[multi_threaded_hmap]" ) {
	std::cout << "Starting concurrent hash map test with " << std::thread::hardware_concurrency() << " threads running concurrently" << std::endl;

	stratus::ConcurrentHashMap<int, int> map;

	REQUIRE(map.Size() == 0);
	REQUIRE(map.Empty() == true);

	const size_t numThreads = std::thread::hardware_concurrency();
	const size_t elementsPerThread = 100000;
	const size_t totalElements = numThreads * elementsPerThread;

	auto start = std::chrono::system_clock::now();
	std::vector<std::thread> threads;

	// Fire off all the threads
	for (size_t i = 0; i < numThreads; ++i) {
		const auto threadNum = i;
		threads.push_back(std::thread([&map, elementsPerThread, threadNum]() {
			// Offset a certain amount so that each thread will be writing distinct keys
			const int beginElem = elementsPerThread * threadNum;
			const int endElem = elementsPerThread + beginElem;
			for (int j = beginElem; j < endElem; ++j) {
				map.InsertIfAbsent(std::make_pair(j, j + 1 + threadNum));
				REQUIRE(map.Contains(j));
				REQUIRE(map.Find(j)->second == (j + 1 + threadNum));
			}
		}));
	}

	// Wait for each to finish before we end the timer
	for (auto & th : threads) th.join();
	auto end = std::chrono::system_clock::now();

	REQUIRE(map.Size() == totalElements);

	std::cout << "Multi threaded test completed in " << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " msec" << std::endl;
	std::cout << "Completed a total of at least " << totalElements << " read/write operations" << std::endl << std::endl;
}