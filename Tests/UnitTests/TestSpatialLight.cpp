#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>
#include <chrono>
#include <any>

#include "StratusLight.h"

TEST_CASE("Stratus Spatial Light Test", "[stratus_spatial_light_test]") {
	std::cout << "Beginning spatial light test" << std::endl;

	stratus::SpatialLightMap lights(256);
	REQUIRE(lights.Size() == 0);

	REQUIRE(lights.CalculateTileIndex(0, 0) == lights.ConvertWorldPosToTileIndex(glm::vec3(0.0f)));
	REQUIRE(lights.CalculateTileIndex(1, 0) == lights.ConvertWorldPosToTileIndex(glm::vec3(256.0f, 0.0f, 0.0f)));
	REQUIRE(lights.CalculateTileIndex(0, 1) == lights.ConvertWorldPosToTileIndex(glm::vec3(0.0f, 0.0f, 256.0f)));
	REQUIRE(lights.CalculateTileIndex(1, 1) == lights.ConvertWorldPosToTileIndex(glm::vec3(256.0f, 0.0f, 256.0f)));

	REQUIRE(lights.CalculateTileIndex(-1, 0) == lights.ConvertWorldPosToTileIndex(glm::vec3(-256.0f, 0.0f, 0.0f)));
	REQUIRE(lights.CalculateTileIndex(0, -1) == lights.ConvertWorldPosToTileIndex(glm::vec3(0.0f, 0.0f, -256.0f)));
	REQUIRE(lights.CalculateTileIndex(-1, -1) == lights.ConvertWorldPosToTileIndex(glm::vec3(-256.0f, 0.0f, -256.0f)));

	std::cout << std::endl;
}