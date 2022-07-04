#include <catch2/catch_all.hpp>
#include "StratusUtils.h"
#include <iostream>

TEST_CASE( "Testing replace", "[replace_test]" ) {
    std::cout << "Beginning stratus::Utils replace test" << std::endl;

    std::string source = "";
    REQUIRE(stratus::ReplaceFirst(source, "1", "one") == false);
    REQUIRE(source == "");
}