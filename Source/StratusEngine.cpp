#include "StratusEngine.h"
#include "StratusThread.h"
#include "StratusLog.h"
#include "StratusMaterial.h"
#include <atomic>
#include <mutex>

    // // Engine class which handles initializing all core engine subsystems and helps 
    // // keep everything in sync during frame updates
    // class Engine {
    //     // Performs first-time engine and subsystem init
    //     Engine(const EngineInitParams &);

    // public:
    //     // Performs first-time start up and then begins the main loop
    //     // returns: true if EngineMain should be called again, false otherwise
    //     friend bool EngineMain(Application * app, const int numArgs, const char ** args);

    //     // Creates a new application and calls EngineMain
    //     template<typename E>
    //     friend void EngineBoot(const int numArgs, const char ** args) {
    //         // Ensure E is derived from Application
    //         static_assert(std::is_base_of<Application, E>::value);
    //         while (true) {
    //             std::unique_ptr<Application> app(new E());
    //             if (!EngineMain(app.get(), numArgs, args)) break;
    //         }
    //     }

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

    //     // Pre-initialization for things like CommandLine, Log, Filesystem
    //     void PreInitialize();
    //     // Initialize rest of the system
    //     void Initialize();
    //     // Should be called before Shutdown()
    //     void BeginShutdown();
    //     // Begins shutdown sequence for engine and all core subsystems
    //     void ShutDown();
    //     // Processes the next full system frame, including rendering. Returns false only
    //     // if the main engine loop should stop.
    //     SystemStatus Frame();

    //     // Main thread is where both engine + application run
    //     Thread * GetMainThread() const;

    // private:
    //     // Global engine instance - should only be set by EngineMain function
    //     static Engine * _instance;
    //     EngineStatistics _stats;
    //     EngineInitParams _params;
    //     Thread * _main;
    //     std::atomic<bool> _isInitializing{false};
    //     std::atomic<bool> _isShuttingDown{false};

    //     // Set of locks for synchronizing different engine operations
    //     mutable std::shared_mutex _startupShutdown;
    //     mutable std::shared_mutex _mainLoop;

namespace stratus {
    Engine * Engine::_instance = nullptr;

    #define CHECK_IS_MAIN_THREAD() assert(Thread::Current() == *Engine::Instance()->GetMainThread())

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
        params.application = app;
        Thread mainThread("Main", false);

        // Delete the instance in case it's left over from a previous run
        delete Engine::_instance;
        Engine::_instance = new Engine(params);

        // Perform first-time initialize
        const auto init = []() {
            Engine::Instance()->PreInitialize();
            Engine::Instance()->Initialize();
        };
        mainThread.Queue(init);
        mainThread.Dispatch();

        // Set up shut down sequence
        const auto shutdown = []() {
            Engine::Instance()->BeginShutDown();
            Engine::Instance()->ShutDown();
        };

        // Enter main loop
        std::atomic<SystemStatus> status(SystemStatus::SYSTEM_CONTINUE);
        const auto runFrame = [&status]() {
            status.store(Engine::Instance()->Frame());
        };

        volatile bool shouldRestart = false;
        volatile bool running = true;
        while (running) {
            mainThread.Queue(runFrame);
            // No need to synchronize since mainThread uses this thread's context
            mainThread.Dispatch();

            // Check the system status message and decide what to do next
            switch (status.load()) {
            case SystemStatus::SYSTEM_RESTART:
                shouldRestart = true;
                // fall down to next
            case SystemStatus::SYSTEM_SHUTDOWN:
                running = false;
                break;
            case SystemStatus::SYSTEM_PANIC:
                exit(-1);
            default:
                continue;
            }
        }

        // Queue and dispatch shutdown sequence
        mainThread.Queue(shutdown);
        mainThread.Dispatch();

        // If this is true the boot function will call EngineMain again
        return shouldRestart;
    }

    Engine::Engine(const EngineInitParams& params)
        : _params(params) {}

    bool Engine::IsInitializing() const {
        return _isInitializing.load();
    }

    bool Engine::IsShuttingDown() const {
        return _isShuttingDown.load();
    }

    uint64_t Engine::FrameCount() const {
        return _stats.currentFrame;
    }

    double Engine::LastFrameTimeSeconds() const {
        return _stats.lastFrameTimeSeconds;
    }

    void Engine::PreInitialize() {
        if (IsInitializing()) {
            throw std::runtime_error("Engine::PreInitialize called twice");
        }
        std::unique_lock<std::shared_mutex> ul(_startupShutdown);
        _isInitializing.store(true);
        
        // Pull the main thread
        _main = &Thread::Current();

        _InitLog();
    }

    void Engine::Initialize() {
        if (!IsInitializing()) {
            throw std::runtime_error("Engine::PreInitialize not called");
        }
        std::unique_lock<std::shared_mutex> ul(_startupShutdown);

        STRATUS_LOG << "Engine initializing" << std::endl;

        _InitMaterialManager();

        // Initialize application last
        _params.application->Initialize();

        STRATUS_LOG << "Initialization complete" << std::endl;
        _isInitializing.store(false);
    }

    void Engine::_InitLog() {
        delete Log::_instance;
        Log::_instance = new Log();
    }

    void Engine::_InitMaterialManager() {
        delete MaterialManager::_instance;
        MaterialManager::_instance = new MaterialManager();
    }

    // Should be called before Shutdown()
    void Engine::BeginShutDown() {
        if (IsShuttingDown()) return;
        _isShuttingDown.store(true);
    }

    // Begins shutdown sequence for engine and all core subsystems
    void Engine::ShutDown() {
        if (!IsShuttingDown()) {
            throw std::runtime_error("Must call Engine::BeginShutdown before Engine::ShutDown");
        }

        STRATUS_LOG << "Engine shutting down" << std::endl;

        // Application should shut down first
        _params.application->ShutDown();

        // Shutdown logic....
    }

    // Processes the next full system frame, including rendering. Returns false only
    // if the main engine loop should stop.
    SystemStatus Engine::Frame() {
        // Validate
        CHECK_IS_MAIN_THREAD();

        if (IsInitializing()) return SystemStatus::SYSTEM_CONTINUE;
        if (IsShuttingDown()) return SystemStatus::SYSTEM_SHUTDOWN;

        std::unique_lock<std::shared_mutex> ul(_mainLoop);

        // Calculate new frame time
        const auto end = std::chrono::system_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - _stats.prevFrameStart).count();
        const double deltaSeconds = duration / 1000.0;
        const double frameRate = 1.0 / deltaSeconds;

        // Make sure we haven't exceeded the max frame rate
        if (frameRate > _params.maxFrameRate) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            return SystemStatus::SYSTEM_CONTINUE;
        }

        // Frame counter should always be +1 for each valid frame
        ++_stats.currentFrame;

        // Update prev frame start to be the beginning of this current frame
        _stats.prevFrameStart = end;

        // Other logic.....

        return _params.application->Update(deltaSeconds);
    }

    // Main thread is where both engine + application run
    Thread * Engine::GetMainThread() const {
        return _main;
    }
}