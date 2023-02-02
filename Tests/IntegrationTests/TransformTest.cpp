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
                // std::cout << "FAILED" << std::endl;
                // std::cout << m1 << std::endl;
                // std::cout << m2 << std::endl;
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

    static bool failed;
    failed = false;

    class TransformTest : public stratus::Application {
    public:
        virtual ~TransformTest() = default;

        stratus::Entity2Ptr prev;

        const char * GetAppName() const override {
            return "TransformTest";
        }

        void CreateEntitiesRecursive(stratus::Entity2Ptr root, int maxDepth) {
            if (maxDepth < 1) return;

            for (int i = 0; i < 5; ++i) {
                auto ptr = stratus::CreateTransformEntity();
                entities.push_back(ptr);

                auto local = ptr->Components().GetComponent<stratus::LocalTransformComponent>().component;
                auto scale = glm::vec3(RandFloat(1000), RandFloat(1000), RandFloat(1000));
                auto rotate = stratus::Rotation(
                    stratus::Degrees(RandFloat(90)), stratus::Degrees(RandFloat(90)), stratus::Degrees(RandFloat(90))  
                );
                auto position = glm::vec3(RandFloat(1000), RandFloat(1000), RandFloat(1000));

                local->SetLocalTransform(scale, rotate, position);
                if (root) root->AttachChildNode(ptr);

                CreateEntitiesRecursive(ptr, maxDepth - 1);
            }
        }

        int CalculateNumNodes(stratus::Entity2Ptr root) {
            int number = root->GetChildNodes().size();
            for (auto ptr : root->GetChildNodes()) {
                number += CalculateNumNodes(ptr);
            }
            return number;
        }

        bool Initialize() override {
            CreateEntitiesRecursive(nullptr, 4);
            int treeSize = 0;
            for (auto ptr : entities) {
                if (ptr->GetParentNode() == nullptr) {
                    treeSize = treeSize + 1 + CalculateNumNodes(ptr);
                }
            }
            STRATUS_LOG << "TREE SIZE: " << treeSize << std::endl;
            for (auto ptr : entities) {
                if (ptr->GetParentNode() == nullptr) {
                    INSTANCE(EntityManager)->AddEntity(ptr);
                }
            }
            return true; // success
        }

        void PerformEntityCopyTest() {
            if (firstUpdate) {
                if (entities.size() == 0) failed = true;
                for (auto ptr : entities) {
                    auto copy = ptr->Copy();
                    if (copy.get() == ptr.get()) failed = true;

                    auto c1 = ptr->Components().GetAllComponents();
                    for (const auto& p : c1) {
                        auto c2 = copy->Components().GetComponentByName(p.component->TypeName());
                        if (p.component == nullptr || c2.component == nullptr || p.component == c2.component) {

                            failed = true;
                            return;
                        }
                    }

                    auto local1 = ptr->Components().GetComponent<stratus::LocalTransformComponent>().component;
                    auto global1 = ptr->Components().GetComponent<stratus::GlobalTransformComponent>().component;

                    auto local2 = copy->Components().GetComponent<stratus::LocalTransformComponent>().component;
                    auto global2 = copy->Components().GetComponent<stratus::GlobalTransformComponent>().component;

                    if (!CheckEquals(local1->GetLocalTransform(), local2->GetLocalTransform()) ||
                        !CheckEquals(global1->GetGlobalTransform(), global2->GetGlobalTransform())) {

                        failed = true;
                        return;
                    }
                }
            }

            firstUpdate = false;
        }

        stratus::SystemStatus Update(const double deltaSeconds) override {
            STRATUS_LOG << "Frame #" << Engine()->FrameCount() << std::endl;

            if (Engine()->FrameCount() >= 30) return stratus::SystemStatus::SYSTEM_SHUTDOWN;

            if (previousUpdated) {
                previousUpdated = false;
                for (auto ptr : entities) {
                    auto local = ptr->Components().GetComponent<stratus::LocalTransformComponent>().component;
                    auto global = ptr->Components().GetComponent<stratus::GlobalTransformComponent>().component;
                    if (!local->ChangedWithinLastFrame() || !global->ChangedWithinLastFrame()) failed = true;
                }
            }

            if (nextFrameUpdate == Engine()->FrameCount()) {
                PerformEntityCopyTest();

                nextFrameUpdate += 2;
                previousUpdated = true;
                for (auto ptr : entities) {
                    auto local = (stratus::LocalTransformComponent *)ptr->Components().GetComponent<stratus::LocalTransformComponent>().component;
                    auto global = (stratus::GlobalTransformComponent *)ptr->Components().GetComponent<stratus::GlobalTransformComponent>().component;
                    if (local->ChangedThisFrame() || global->ChangedThisFrame()) failed = true;

                    auto parentGlobal = ptr->GetParentNode() != nullptr
                        ? (stratus::GlobalTransformComponent *)ptr->GetParentNode()->Components().GetComponent<stratus::GlobalTransformComponent>().component
                        : nullptr;
                    
                    if (parentGlobal) {
                        if (!CheckEquals(global->GetGlobalTransform(), parentGlobal->GetGlobalTransform() * local->GetLocalTransform())) {
                            failed = true;
                        }
                    }
                    else {
                        if (!CheckEquals(global->GetGlobalTransform(), local->GetLocalTransform())) {
                            failed = true;
                        }
                    }

                    auto scale = glm::vec3(RandFloat(1000), RandFloat(1000), RandFloat(1000));
                    auto rotate = stratus::Rotation(
                        stratus::Degrees(RandFloat(90)), stratus::Degrees(RandFloat(90)), stratus::Degrees(RandFloat(90))  
                    );
                    auto position = glm::vec3(RandFloat(1000), RandFloat(1000), RandFloat(1000));

                    local->SetLocalTransform(scale, rotate, position);
                }
            }

            return stratus::SystemStatus::SYSTEM_CONTINUE;
        }

        void Shutdown() override {
        }

        bool firstUpdate = true;
        bool previousUpdated = true;
        uint64_t nextFrameUpdate = 2;
        std::vector<stratus::Entity2Ptr> entities;
    };

    STRATUS_INLINE_ENTRY_POINT(TransformTest, numArgs, argList);

    REQUIRE_FALSE(failed);
}