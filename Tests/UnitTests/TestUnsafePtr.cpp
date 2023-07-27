#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>
#include <chrono>

#include "StratusPointer.h"

TEST_CASE( "Stratus Stack Allocators Test", "[stratus_pointer_test]" ) {
	std::cout << "Beginning stratus::UnsafePtr tests" << std::endl;

	auto ptr = stratus::UnsafePtr<int>();
	REQUIRE(ptr == nullptr);
	REQUIRE(ptr.RefCount() == 0);

	ptr = stratus::MakeUnsafe<int>(42);
	REQUIRE(ptr != nullptr);
	REQUIRE(ptr.RefCount() == 1);
	REQUIRE(*ptr == 42);

    static bool destroyed;
    destroyed = false;
    struct Test {
        int value;

        Test(int value) : value(value) {}

        ~Test() {
            destroyed = true;
        }
    };

    auto tmp = new Test(81);
    auto testPtr = stratus::UnsafePtr<Test>(tmp);
    REQUIRE(testPtr != nullptr);
    REQUIRE(testPtr.RefCount() == 1);
    REQUIRE(testPtr.Get() == tmp);
    REQUIRE(testPtr->value == 81);

    auto testPtr2 = testPtr;
    REQUIRE(testPtr2 == testPtr);
    REQUIRE(testPtr2.Get() == tmp);
    REQUIRE(testPtr2.RefCount() == 2);
    REQUIRE(testPtr->value == testPtr2->value);

    testPtr2.Reset();
    REQUIRE(testPtr2 == nullptr);
    REQUIRE(testPtr2.RefCount() == 0);
    REQUIRE(testPtr.RefCount() == 1);
    REQUIRE(destroyed == false);

    testPtr.Reset();
    REQUIRE(testPtr == nullptr);
    REQUIRE(testPtr.RefCount() == 0);
    REQUIRE(destroyed == true);
}