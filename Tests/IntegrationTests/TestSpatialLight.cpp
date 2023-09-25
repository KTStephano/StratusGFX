#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>
#include <chrono>
#include <any>

#include "StratusLight.h"
#include "StratusEngine.h"
#include "StratusApplication.h"
#include "StratusLog.h"
#include "StratusCommon.h"
#include "IntegrationMain.h"

TEST_CASE("Stratus Spatial Light Test", "[stratus_spatial_light_test]") {
	std::cout << "Beginning spatial light test" << std::endl;

	{
		stratus::SpatialLightMap lights(256);
		REQUIRE(lights.Size() == 0);

		REQUIRE(lights.CalculateTileIndex(0, 0) == lights.ConvertWorldPosToTileIndex(glm::vec3(0.0f)));
		REQUIRE(lights.CalculateTileIndex(1, 0) == lights.ConvertWorldPosToTileIndex(glm::vec3(256.0f, 0.0f, 0.0f)));
		REQUIRE(lights.CalculateTileIndex(0, 1) == lights.ConvertWorldPosToTileIndex(glm::vec3(0.0f, 0.0f, 256.0f)));
		REQUIRE(lights.CalculateTileIndex(1, 1) == lights.ConvertWorldPosToTileIndex(glm::vec3(256.0f, 0.0f, 256.0f)));

		REQUIRE(lights.CalculateTileIndex(-1, 0) == lights.ConvertWorldPosToTileIndex(glm::vec3(-256.0f, 0.0f, 0.0f)));
		REQUIRE(lights.CalculateTileIndex(0, -1) == lights.ConvertWorldPosToTileIndex(glm::vec3(0.0f, 0.0f, -256.0f)));
		REQUIRE(lights.CalculateTileIndex(-1, -1) == lights.ConvertWorldPosToTileIndex(glm::vec3(-256.0f, 0.0f, -256.0f)));
	}

    static bool testsSucceeded = false;
    testsSucceeded = true;

	class SpatialLightTest : public stratus::Application {
    public:
        virtual ~SpatialLightTest() = default;

        const char * GetAppName() const override {
            return "SpatialLightTest";
        }

        virtual bool Initialize() override {
            stratus::SpatialLightMap lights(256);
            REQUIRE(lights.Size() == 0);

            auto light = stratus::LightPtr(new stratus::VirtualPointLight());
            light->SetPosition(glm::vec3(0.0f));

            lights.Insert(light);
            if (lights.Size() != 1 || !lights.Contains(light)) {
                testsSucceeded = false;
            }

            lights.Insert(light);
            if (lights.Size() != 1 || !lights.Contains(light)) {
                testsSucceeded = false;
            }

            auto nearest = lights.GetNearestTiles(glm::vec3(0.0f), 1);
            if (nearest.size() != 1 ||
                nearest[0].Lights().size() != 1 ||
                nearest[0].Lights().find(light) == nearest[0].Lights().end()) {
                testsSucceeded = false;
            }

            nearest = lights.GetNearestTiles(glm::vec3(1024.0f), 1);
            if (nearest.size() != 0) {
                testsSucceeded = false;
            }

            auto light2 = stratus::LightPtr(new stratus::VirtualPointLight());
            light2->SetPosition(glm::vec3(1024.0f));
            lights.Insert(light2);

            nearest = lights.GetNearestTiles(glm::vec3(1024.0f), 1);
            if (nearest.size() != 1 ||
                nearest[0].Lights().size() != 1 ||
                nearest[0].Lights().find(light2) == nearest[0].Lights().end()) {
                testsSucceeded = false;
            }

            light->SetPosition(glm::vec3(1024.0f, 0.0f, 1024.0f - 256.0f));
            lights.Insert(light);

            if (lights.Size() != 2) {
                testsSucceeded = false;
            }

            nearest = lights.GetNearestTiles(glm::vec3(1024.0f), 1);
            if (nearest.size() != 2) {
                testsSucceeded = false;
            }
            else {
                bool succeeded1 = false;
                for (auto& container : nearest) {
                    if (container.Lights().find(light) != container.Lights().end()) {
                        succeeded1 = true;
                        break;
                    }
                }

                bool succeeded2 = false;
                for (auto& container : nearest) {
                    if (container.Lights().find(light2) != container.Lights().end()) {
                        succeeded2 = true;
                        break;
                    }
                }

                testsSucceeded = testsSucceeded && succeeded1 && succeeded2;
            }

            return true; // success
        }

        virtual stratus::SystemStatus Update(const double deltaSeconds) override {
            STRATUS_LOG << "Successfully entered SpatialLightTest::Update! Delta seconds = " << deltaSeconds << std::endl;
            return stratus::SystemStatus::SYSTEM_SHUTDOWN;
        }

        virtual void Shutdown() override {
            STRATUS_LOG << "Successfully entered SpatialLightTest::ShutDown()" << std::endl;
        }
    };

    STRATUS_INLINE_ENTRY_POINT(SpatialLightTest, numArgs, argList);

    REQUIRE(testsSucceeded);

	std::cout << std::endl;
}