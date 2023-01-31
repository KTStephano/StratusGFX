#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>
#include <random>

#include "StratusEngine.h"
#include "StratusApplication.h"
#include "StratusLog.h"
#include "StratusCommon.h"
#include "IntegrationMain.h"
#include "StratusEntityManager.h"
#include "StratusEntityProcess.h"
#include "StratusEntity2.h"
#include "StratusEntityCommon.h"
#include "StratusTransformComponent.h"

static bool CheckEquals(const glm::mat4& m1, const glm::mat4& m2) {
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
            const auto diff = std::fabs(m1[x][y] - m2[x][y]);
            if (diff > std::numeric_limits<float>::epsilon()) {
                std::cout << "FAILED" << std::endl;
                std::cout << m1 << std::endl;
                std::cout << m2 << std::endl;
                return false;
            }
        }
    }
    return true;
}

static float RandFloat(uint32_t mod) {
    return float(rand() % mod);
}

TEST_CASE("Stratus Transform Test", "[stratus_transform_test]") {
    srand((unsigned)time(nullptr));

    std::cout << "StratusTransformTest" << std::endl;

    stratus::LocalTransformComponent c;

    REQUIRE(CheckEquals(c.GetLocalTransform(), glm::mat4(1.0f)));

    for (int i = 0; i < 1000; ++i) {
        auto scale = glm::vec3(RandFloat(1000), RandFloat(1000), RandFloat(1000));
        auto rotate = stratus::Rotation(
            stratus::Degrees(RandFloat(90)), stratus::Degrees(RandFloat(90)), stratus::Degrees(RandFloat(90))  
        );
        auto position = glm::vec3(RandFloat(1000), RandFloat(1000), RandFloat(1000));

        auto S = glm::mat4(1.0f);
        stratus::matScale(S, scale);
        auto R = rotate.asMat4();
        auto T = glm::mat4(1.0f);
        stratus::matTranslate(T, position);

        c.SetLocalTransform(scale, rotate, position);

        REQUIRE(CheckEquals(c.GetLocalTransform(), T * R * S));
    }
}