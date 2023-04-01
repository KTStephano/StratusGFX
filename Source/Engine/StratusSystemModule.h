#pragma once

#include "StratusSystemStatus.h"

namespace stratus {

    #define SYSTEM_MODULE_CLASS(name)                        \
        class name : public SystemModule {                   \
            friend class Engine;                             \
            friend struct EngineModuleInit;                  \
            name();                                          \
            static name *& Instance_() {                     \
                static name * instance = nullptr;            \
                return instance;                             \
            }                                                \
        public:                                              \
            const char * Name() const override {             \
                return #name;                                \
            }                                                \
            static name * Instance() { return Instance_(); }

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