#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>

#include "StratusHandle.h"

struct S {};

TEST_CASE( "Stratus Handle Test", "[stratus_handle_test]" ) {
    std::cout << "Beginning stratus::Handle test" << std::endl;

    typedef stratus::Handle<S> TestHandle;

    REQUIRE(TestHandle() == TestHandle::Null());
    REQUIRE(TestHandle::Null() == TestHandle::Null());
    REQUIRE(TestHandle::Null() != TestHandle::NextHandle());
    REQUIRE(!TestHandle::Null());
    REQUIRE(TestHandle::NextHandle());

    std::unordered_set<TestHandle> set;
    const size_t numHandleElems = 100;
    for (size_t i = 0; i < numHandleElems; ++i) {
        set.insert(TestHandle::NextHandle());
    }
    REQUIRE(set.size() == numHandleElems);
}