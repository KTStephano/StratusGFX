#include <catch2/catch_all.hpp>
#include <iostream>
#include <unordered_set>

#include "StratusEngine.h"
#include "StratusApplication.h"
#include "StratusLog.h"
#include "StratusCommon.h"
#include "IntegrationMain.h"

TEST_CASE( "Stratus Startup Shutdown", "[stratus_startup_shutdown]" ) {
    static bool reachedInitialize = false;
    static bool reachedUpdate = false;
    static bool reachedShutdown = false;

    class StartupShutdown : public stratus::Application {
    public:
        virtual ~StartupShutdown() = default;

        const char * GetAppName() const override {
            return "StartupShutdown";
        }

        virtual bool Initialize() override {
            reachedInitialize = true;
            return true; // success
        }

        virtual stratus::SystemStatus Update(const double deltaSeconds) override {
            reachedUpdate = true;
            STRATUS_LOG << "Successfully entered StartupShutdown::Update! Delta seconds = " << deltaSeconds << std::endl;
            return stratus::SystemStatus::SYSTEM_SHUTDOWN;
        }

        virtual void Shutdown() override {
            reachedShutdown = true;
            STRATUS_LOG << "Successfully entered StartupShutdown::ShutDown()" << std::endl;
        }
    };

    STRATUS_INLINE_ENTRY_POINT(StartupShutdown, numArgs, argList);

    // Make sure we hit all required functions
    REQUIRE(reachedInitialize);
    REQUIRE(reachedUpdate);
    REQUIRE(reachedShutdown);
}