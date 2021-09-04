#include "StratusEngine.h"
#include "StratusThread.h"
#include <atomic>
#include <mutex>

    // // Contains everything the engine needs to perform first-time init
    // struct EngineInitParams {
    //     uint32_t           numCmdArgs;
    //     const char **      cmdArgs;
    //     uint32_t           frameRateCap = 1000;
    // };

    // struct EngineStatistics {
    //     // Each frame update increments this by 1
    //     uint64_t currentFrame = 0;
    //     // Records the time the last frame took to complete - 16.0/1000.0 = 60 fps for example
    //     double lastFrameTimeSeconds;
    //     // Average over all previous frames since engine init
    //     double totalAverageFrameTimeSeconds;
    // };

    // // Engine class which handles initializing all core engine subsystems and helps 
    // // keep everything in sync during frame updates
    // class Engine {
    //     // Performs first-time engine and subsystem init
    //     Engine(const EngineInitParams &);

    // public:
    //     // Performs first-time start up and then begins the main loop
    //     friend void EngineMain(const int numArgs, const char ** args);

    //     // Global engine instance
    //     static Engine * Instance() { return _instance; }

    //     // Checks if the engine has completed its init phase
    //     bool IsInitializing() const;
    //     // True if the engine is performing final shutdown sequence
    //     bool IsShuttingDown() const;
    //     // Returns how many frames the engine has processed since first start
    //     uint64_t FrameCount() const;
    //     // Useful functions for checking current and average frame delta seconds
    //     double LastFrameTimeSeconds() const;
    //     double TotalAverageFrameTimeSeconds() const;

    //     // Pre-initialization for things like CommandLine, Log, Filesystem
    //     void PreInitialize();
    //     // Initialize rest of the system
    //     void Initialize();
    //     // Begins shutdown sequence for engine and all core subsystems
    //     void ShutDown();
    //     // Processes the next full system frame, including rendering. Returns false only
    //     // if the main engine loop should stop.
    //     bool Frame();

    //     // Main thread is where both engine + application run
    //     Thread * GetMainThread() const;

    // private:
    //     // Global engine instance - should only be set by EngineMain function
    //     static Engine * _instance;
    //     EngineStatistics _stats;
    //     EngineInitParams _params;
    //     Thread * _main;

namespace stratus {
    Engine * Engine::_instance = nullptr;

    bool EngineMain(Application * app, const int numArgs, const char ** args) {
        static std::mutex preventMultipleMainCalls;
        std::unique_lock<std::mutex> ul(preventMultipleMainCalls, std::defer_lock);
        if (!ul.try_lock()) {
            throw std::runtime_error("EngineMain already running");
        }

        // Set up engine params and create global Engine instance
        EngineInitParams params;
        params.numCmdArgs = numArgs;
        params.cmdArgs = args;
        // Delete the instance in case it's left over from a previous run
        delete Engine::_instance;
        Engine::_instance = new Engine(params);
        Thread mainThread("Main", false);

        // Perform first-time initialize
        const auto init = []() {
            Engine::Instance()->PreInitialize();
            Engine::Instance()->Initialize();
        };
        mainThread.Queue(init);
        mainThread.DispatchAndSynchronize();

        // Enter main loop
        std::atomic<bool> running(true);
        const auto runFrame = [&running]() {
            running.store(Engine::Instance()->Frame());
        };

        while (running.load()) {
            mainThread.Queue(runFrame);
            // No need to synchronize since mainThread uses this thread's context
            mainThread.Dispatch();
        }

        // If this is true the boot function will call EngineMain again
        return Engine::Instance()->ShouldRestart();
    }

    Engine::Engine(const EngineInitParams& params)
        : _params(params) {}
}