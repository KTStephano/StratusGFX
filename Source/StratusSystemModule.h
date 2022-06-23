#pragma once

#include "StratusSystemStatus.h"

namespace stratus {
    // Interface for consistent initialize/shutdown behavior for modules such as
    // log, resource manager, renderer frontend, etc.
    struct SystemModule {
        virtual ~SystemModule() = default;
        
        virtual const char * Name() const = 0;
        // Return true for success, false for failure
        virtual bool Initialize() = 0;
        // SystemStatus return tells the engine how to proceed
        virtual SystemStatus Update(const double deltaSeconds) = 0;
        // Clean up all resources
        virtual void Shutdown() = 0;
    };
}