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

TEST_CASE( "Stratus Entity Test", "[stratus_entity_test]" ) {
    static constexpr size_t maxEntities = 10;
    static size_t numEntitiesAdded = 0;
    static size_t numEntitiesRemoved = 0;
    static bool processCalled = false;

    class ProcessTest : public stratus::EntityProcess {
        friend class stratus::EntityManager;

        ProcessTest() {
            STRATUS_LOG << "ProcessTest registered successfully" << std::endl;
        }
        
        virtual ~ProcessTest() = default;

        virtual void Process(const double deltaSeconds) override {
            processCalled = true;
            STRATUS_LOG << "Process " << deltaSeconds << std::endl;
        }

        virtual void EntitiesAdded(const std::unordered_set<stratus::Entity2Ptr>& e) override {
            numEntitiesAdded += e.size();
        }

        virtual void EntitiesRemoved(const std::unordered_set<stratus::Entity2Ptr>& e) override {
            numEntitiesRemoved += e.size();
        }
    };

    class EntityTest : public stratus::Application {
    public:
        virtual ~EntityTest() = default;

        stratus::Entity2Ptr prev;

        const char * GetAppName() const override {
            return "EntityTest";
        }

        virtual bool Initialize() override {
            INSTANCE(EntityManager)->RegisterEntityProcess<ProcessTest>();
            return true; // success
        }

        virtual stratus::SystemStatus Update(const double deltaSeconds) override {
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

        virtual void Shutdown() override {
        }
    };

    STRATUS_INLINE_ENTRY_POINT(EntityTest, numArgs, argList);

    REQUIRE(numEntitiesAdded == maxEntities);
    REQUIRE(numEntitiesRemoved == maxEntities);
    REQUIRE(processCalled);
}