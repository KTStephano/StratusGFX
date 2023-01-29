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

ENTITY_COMPONENT_STRUCT(ExampleComponent)
    const void * ptr = nullptr;

    ExampleComponent(const void * ptr) : ptr(ptr) {
        STRATUS_LOG << "ExampleComponent created" << std::endl;
    }

    virtual ~ExampleComponent() = default;
};

TEST_CASE("Stratus Entity Test", "[stratus_entity_test]") {
    static constexpr size_t maxEntities = 10;
    static size_t numEntitiesAdded;
    static size_t numEntitiesRemoved;
    static size_t numComponentsAdded;
    static bool processCalled;
    static bool componentsEnabledDisabledCalled;
    static bool processRanIntoIssues;

    numEntitiesAdded = 0;
    numEntitiesRemoved = 0;
    numComponentsAdded = 0;
    processCalled = false;
    componentsEnabledDisabledCalled = false;
    processRanIntoIssues = false;


    struct ProcessTest : public stratus::EntityProcess {
        int randomEntityIndex;

        ProcessTest() {
            STRATUS_LOG << "ProcessTest registered successfully" << std::endl;
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist(0, maxEntities - 1); // range [0, maxEntities - 1] inclusive
            randomEntityIndex = dist(rng);
            std::cout << "Using " << randomEntityIndex << std::endl;
        }
        
        virtual ~ProcessTest() = default;

        void Process(const double deltaSeconds) override {
            if (ptrs.size() >= randomEntityIndex) {
                int i = 0;
                for (auto ptr : ptrs) {
                    if (i == randomEntityIndex) {
                        disabledComponent = ptr;
                        ptr->Components().DisableComponent<ExampleComponent>();
                    }
                    else {
                        if (ptr->Components().GetComponent<ExampleComponent>().status != stratus::EntityComponentStatus::COMPONENT_ENABLED) {
                            processRanIntoIssues = true;
                        }
                    }
                    ++i;
                }
            }
            processCalled = true;
            STRATUS_LOG << "Process " << deltaSeconds << std::endl;
        }

        void EntitiesAdded(const std::unordered_set<stratus::Entity2Ptr>& e) override {
            numEntitiesAdded += e.size();
            for (stratus::Entity2Ptr ptr : e) {
                ptrs.push_back(ptr);
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
                if (entry.first->Components().GetComponent<ExampleComponent>().component->ptr != (const void *)this) continue;
                // Make sure the component we added actually shows up in the array
                for (stratus::Entity2Component * c : components) {
                    if (c->TypeName() == ExampleComponent::STypeName()) {
                        numComponentsAdded += 1;
                    }
                }
            }
        }

        void EntityComponentsEnabledDisabled(const std::unordered_set<stratus::Entity2Ptr>& changed) override {
            componentsEnabledDisabledCalled = true;
            for (auto ptr : changed) {
                if (ptr != disabledComponent || 
                    ptr->Components().GetComponent<ExampleComponent>().status != stratus::EntityComponentStatus::COMPONENT_DISABLED) {
                    
                    processRanIntoIssues = true;
                }
            }
        }

        stratus::Entity2Ptr disabledComponent;
        std::vector<stratus::Entity2Ptr> ptrs;
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
    REQUIRE(componentsEnabledDisabledCalled);
    REQUIRE_FALSE(processRanIntoIssues);
}