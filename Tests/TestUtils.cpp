#include <catch2/catch_all.hpp>
#include "StratusUtils.h"
#include <iostream>

TEST_CASE( "Testing replace", "[replace_test]" ) {
    std::cout << "Beginning stratus::Utils replace test" << std::endl;

    std::string source = "";
    REQUIRE(stratus::ReplaceFirst(source, "1", "one") == false);
    REQUIRE(source == "");

    source = "1";
    REQUIRE(stratus::ReplaceFirst(source, "1", "one") == true);
    REQUIRE(source == "one");

    source = "1 2 3 4 1 5 6 1 7 8";
    REQUIRE(stratus::ReplaceFirst(source, "1", "one") == true);
    REQUIRE(source == "one 2 3 4 1 5 6 1 7 8");

    source = "1 2 3 4 1 5 6 1 7 8";
    REQUIRE(stratus::ReplaceAll(source, "1", "one") == true);
    REQUIRE(source == "one 2 3 4 one 5 6 one 7 8");
}