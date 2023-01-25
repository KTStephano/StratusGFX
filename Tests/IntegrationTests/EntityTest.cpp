#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>

#include "StratusEngine.h"
#include "StratusApplication.h"
#include "StratusLog.h"
#include "StratusCommon.h"
#include "IntegrationMain.h"
#include "StratusEntityManager.h"
#include "StratusEntityProcess.h"
#include "StratusEntity2.h"
#include "StratusEntityCommon.h"

ENTITY_COMPONENT_STRUCT(ExampleComponent)
    const void * ptr = nullptr;

    ExampleComponent(const void * ptr) : ptr(ptr) {
        STRATUS_LOG << "ExampleComponent created" << std::endl;
    }

    virtual ~ExampleComponent() = default;
};

TEST_CASE("Stratus Entity Test", "[stratus_entity_test]") {
    static constexpr size_t maxEntities = 10;
    static size_t numEntitiesAdded = 0;
    static size_t numEntitiesRemoved = 0;
    static size_t numComponentsAdded = 0;
    static bool processCalled = false;

    struct ProcessTest : public stratus::EntityProcess {
        ProcessTest() {
            STRATUS_LOG << "ProcessTest registered successfully" << std::endl;
        }
        
        virtual ~ProcessTest() = default;

        void Process(const double deltaSeconds) override {
            processCalled = true;
            STRATUS_LOG << "Process " << deltaSeconds << std::endl;
        }

        void EntitiesAdded(const std::unordered_set<stratus::Entity2Ptr>& e) override {
            numEntitiesAdded += e.size();
            for (stratus::Entity2Ptr ptr : e) {
                ptr->Components().AttachComponent<ExampleComponent>((const void *)this);
            }
        }

        void EntitiesRemoved(const std::unordered_set<stratus::Entity2Ptr>& e) override {
            numEntitiesRemoved += e.size();
        }

        void EntityComponentsAdded(const std::unordered_map<stratus::Entity2Ptr, std::vector<stratus::Entity2Component*>>& added) override {
            for (auto entry : added) {
                // Don't process if we've handled it before
                if (seen.find(entry.first) != seen.end()) continue;
                seen.insert(entry.first);
                auto components = entry.second;
                // If it doesn't have our component then don't process
                if (!entry.first->Components().ContainsComponent<ExampleComponent>()) continue;
                // If the ptr we set is invalid then don't process
                if (entry.first->Components().GetComponent<ExampleComponent>()->ptr != (const void *)this) continue;
                // Make sure the component we added actually shows up in the array
                for (stratus::Entity2Component * c : components) {
                    if (c->TypeName() == ExampleComponent::STypeName()) {
                        numComponentsAdded += 1;
                    }
                }
            }
        }

        std::unordered_set<stratus::Entity2Ptr> seen;
    };

    class EntityTest : public stratus::Application {
    public:
        virtual ~EntityTest() = default;

        stratus::Entity2Ptr prev;

        const char * GetAppName() const override {
            return "EntityTest";
        }

        bool Initialize() override {
            INSTANCE(EntityManager)->RegisterEntityProcess<ProcessTest>();
            return true; // success
        }

        stratus::SystemStatus Update(const double deltaSeconds) override {
            STRATUS_LOG << "Frame #" << Engine()->FrameCount() << std::endl;


            if (Engine()->FrameCount() > (maxEntities + 1)) return stratus::SystemStatus::SYSTEM_SHUTDOWN;

            if (prev) {
                INSTANCE(EntityManager)->RemoveEntity(prev);
                prev.reset();
            }

            if (Engine()->FrameCount() <= maxEntities) {
                prev = stratus::Entity2::Create();
                INSTANCE(EntityManager)->AddEntity(prev);
            }

            return stratus::SystemStatus::SYSTEM_CONTINUE;
        }

        void Shutdown() override {
        }
    };

    STRATUS_INLINE_ENTRY_POINT(EntityTest, numArgs, argList);

    REQUIRE(numEntitiesAdded == maxEntities);
    REQUIRE(numEntitiesRemoved == maxEntities);
    REQUIRE(numComponentsAdded == maxEntities);
    REQUIRE(processCalled);
}