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

#define STRATUS_ENTRY_POINT(ApplicationClass)                              \
    int main(int nargs, char ** args) {                                    \
        stratus::EngineBoot<ApplicationClass>(nargs, (const char **)args); \
        return 0;                                                          \
    }

namespace stratus {
    class Thread;

    // Contains everything the engine needs to perform first-time init
    struct EngineInitParams {
        uint32_t           numCmdArgs;
        const char **      cmdArgs;
        uint32_t           maxFrameRate = 1000;
        Application *      application;
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
        friend bool EngineMain(Application * app, const int numArgs, const char ** args);

        // Creates a new application and calls EngineMain
        template<typename E>
        friend void EngineBoot(const int numArgs, const char** args);

        // Global engine instance
        static Engine * Instance() { return _instance; }

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
        void _InitLog();
        void _InitMaterialManager();

    private:
        // Global engine instance - should only be set by EngineMain function
        static Engine * _instance;
        EngineStatistics _stats;
        EngineInitParams _params;
        Thread * _main;
        std::atomic<bool> _isInitializing{false};
        std::atomic<bool> _isShuttingDown{false};

        // Set of locks for synchronizing different engine operations
        mutable std::shared_mutex _startupShutdown;
        mutable std::shared_mutex _mainLoop;
    };
       
    template<typename E>
    void EngineBoot(const int numArgs, const char** args) {
        // Ensure E is derived from Application
        static_assert(std::is_base_of<Application, E>::value);
        while (true) {
            std::unique_ptr<Application> app(new E());
            if (!EngineMain(app.get(), numArgs, args)) break;
        }
    }
}

#endif //STRATUSGFX_ENGINE_H