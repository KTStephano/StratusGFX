//
// Created by stephano on 2/17/19.
//

#ifndef STRATUSGFX_ENGINE_H
#define STRATUSGFX_ENGINE_H

#include "StratusCommon.h"
#include "StratusHandle.h"
#include "StratusApplication.h"
#include "StratusSystemStatus.h"
#include <shared_mutex>
#include <memory>
#include <atomic>
#include <chrono>
//#include "Renderer.h"

// Useful within an existing main function
#define STRATUS_INLINE_ENTRY_POINT(ApplicationClass, numArgs, argList)     \
    stratus::EngineBoot<ApplicationClass>(numArgs, (const char **)argList)

// Defines both the main function and the engine startup function
#define STRATUS_ENTRY_POINT(ApplicationClass)                              \
    int main(int nargs, char ** args) {                                    \
        STRATUS_INLINE_ENTRY_POINT(ApplicationClass, nargs, args);         \
        return 0;                                                          \
    }

namespace stratus {
    class Thread;

    // Contains everything the engine needs to perform first-time init
    struct EngineInitParams {
        uint32_t           numCmdArgs;
        const char **      cmdArgs;
        uint32_t           maxFrameRate = 1000;
    };

    struct EngineStatistics {
        // Each frame update increments this by 1
        uint64_t currentFrame = 0;
        // Records the time the last frame took to complete - 16.0/1000.0 = 60 fps for example
        double lastFrameTimeSeconds = 0.0;
        std::chrono::system_clock::time_point prevFrameStart = std::chrono::system_clock::now();
    };

    // Engine class which handles initializing all core engine subsystems and helps 
    // keep everything in sync during frame updates
    class Engine {
        // Performs first-time engine and subsystem init
        Engine(const EngineInitParams &);

    public:
        // Performs first-time start up and then begins the main loop
        // returns: true if EngineMain should be called again, false otherwise
        static bool EngineMain(Application * app, const int numArgs, const char ** args);
        
        // Creates a new application and calls EngineMain
        template<typename E>
        friend void EngineBoot(const int numArgs, const char** args);

        // Global engine instance
        static Engine * Instance() { return instance_; }

        // Sets max frame rate
        void SetMaxFrameRate(const uint32_t);
        // Checks if the engine has completed its init phase
        bool IsInitializing() const;
        // True if the engine is performing final shutdown sequence
        bool IsShuttingDown() const;
        // Returns how many frames the engine has processed since first start
        uint64_t FrameCount() const;
        // Useful functions for checking current and average frame delta seconds
        double LastFrameTimeSeconds() const;

        // Pre-initialization for things like CommandLine, Log, Filesystem
        void PreInitialize();
        // Initialize rest of the system
        void Initialize();
        // Should be called before Shutdown()
        void BeginShutDown();
        // Begins shutdown sequence for engine and all core subsystems
        void ShutDown();
        // Processes the next full system frame, including rendering. Returns false only
        // if the main engine loop should stop.
        SystemStatus Frame();

        // Main thread is where both engine + application run
        Thread * GetMainThread() const;

    private:
        void InitLog_();
        void InitInput_();
        void InitGraphicsDriver_();
        void InitEntityManager_();
        void InitApplicationThread_();
        void InitTaskSystem_();
        void InitMaterialManager_();
        void InitResourceManager_();
        void InitWindow_();
        void InitRenderer_();

        template<typename E>
        void DeleteResource_(E *& ptr) {
            delete ptr;
            ptr = nullptr;
        }

        template<typename E>
        void ShutdownResourceAndDelete_(E *& ptr) {
            ptr->Shutdown();
            DeleteResource_(ptr);
        }

    private:
        // Global engine instance - should only be set by EngineMain function
        static Engine * instance_;
        EngineStatistics stats_;
        EngineInitParams _params;
        Thread * main_;
        std::atomic<bool> isInitializing_{false};
        std::atomic<bool> isShuttingDown_{false};

        // Set of locks for synchronizing different engine operations
        mutable std::shared_mutex startupShutdown_;
        mutable std::shared_mutex mainLoop_;
    };
       
    template<typename E>
    void EngineBoot(const int numArgs, const char** args) {
        // Ensure E is derived from Application
        static_assert(std::is_base_of<Application, E>::value);
        while (true) {
            // Engine owns the pointer and will delete it
            Application * app = new E();
            if (!Engine::EngineMain(app, numArgs, args)) break;
        }
    }
}

#endif //STRATUSGFX_ENGINE_H