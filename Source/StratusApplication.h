#pragma once

#include "StratusSystemStatus.h"

namespace stratus {
    // Special interface class which the engine knows is the entry point
    // for the application (e.g. editor or game)
    class Application {
    public:
        virtual ~Application() = default;

        // Perform first-time initialization - true if success, false otherwise
        virtual bool Initialize() = 0;
        // Run a single update for the application (no infinite loops)
        // deltaSeconds = time since last frame
        virtual SystemStatus Update(double deltaSeconds) = 0;
        // Perform any resource cleanup
        virtual void ShutDown() = 0;
    };
}