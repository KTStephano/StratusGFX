#include "StratusEngine.h"
#include "StratusThread.h"
#include "StratusLog.h"
#include "StratusMaterial.h"
#include "StratusResourceManager.h"
#include "StratusWindow.h"
#include "StratusRendererFrontend.h"
#include "StratusApplicationThread.h"
#include "StratusTaskSystem.h"
#include "StratusEntityManager.h"
#include "StratusGraphicsDriver.h"
#include <atomic>
#include <mutex>

namespace stratus {
    Engine * Engine::instance_ = nullptr;

    bool Engine::EngineMain(Application * app, const int numArgs, const char ** args) {
        static std::mutex preventMultipleMainCalls;
        std::unique_lock<std::mutex> ul(preventMultipleMainCalls, std::defer_lock);
        if (!ul.try_lock()) {
            throw std::runtime_error("EngineMain already running");
        }

        // Set up engine params and create global Engine instance
        EngineInitParams params;
        params.numCmdArgs = numArgs;
        params.cmdArgs = args;
        Application::Instance_() = app;

        // Delete the instance in case it's left over from a previous run
        delete Engine::instance_;
        Engine::instance_ = new Engine(params);

        // Pre-initialize to set up things like the ApplicationThread
        Engine::Instance()->PreInitialize();
        ApplicationThread::Instance()->Queue([]() {
            Engine::Instance()->Initialize();
        });
        ApplicationThread::Instance()->DispatchAndSynchronize_();

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
            ApplicationThread::Instance()->Queue(runFrame);
            // No need to synchronize since ApplicationThread uses this thread's context
            ApplicationThread::Instance()->Dispatch_();

            // Check the system status message and decide what to do next
            switch (status.load()) {
            case SystemStatus::SYSTEM_RESTART:
                shouldRestart = true;
                // fall down to next
            case SystemStatus::SYSTEM_SHUTDOWN:
                running = false;
                break;
            case SystemStatus::SYSTEM_PANIC:
                std::cerr << "Critical error - exiting immediately" << std::endl;
                exit(-1);
            default:
                continue;
            }
        }

        // Queue and dispatch shutdown sequence
        ApplicationThread::Instance()->Queue(shutdown);
        ApplicationThread::Instance()->Dispatch_();

        // If this is true the boot function will call EngineMain again
        return shouldRestart;
    }

    Engine::Engine(const EngineInitParams& params)
        : _params(params) {}

    bool Engine::IsInitializing() const {
        return isInitializing_.load();
    }

    bool Engine::IsShuttingDown() const {
        return isShuttingDown_.load();
    }

    void Engine::SetMaxFrameRate(const uint32_t rate) {
        std::unique_lock<std::shared_mutex> ul(mainLoop_);
        _params.maxFrameRate = std::max<uint32_t>(rate, 30);
    }

    uint64_t Engine::FrameCount() const {
        return stats_.currentFrame;
    }

    double Engine::LastFrameTimeSeconds() const {
        return stats_.lastFrameTimeSeconds;
    }

    void Engine::PreInitialize() {
        if (IsInitializing()) {
            throw std::runtime_error("Engine::PreInitialize called twice");
        }
        std::unique_lock<std::shared_mutex> ul(startupShutdown_);
        isInitializing_.store(true);

        InitLog_();
        InitApplicationThread_();

        // Pull the main thread
        main_ = ApplicationThread::Instance()->thread_.get();
    }

    struct EngineModuleInit {
        template<typename E>
        static void InitializeEngineModule(E * instance, const bool log) {
            if (log) {
                STRATUS_LOG << "Initializing " << instance->Name() << std::endl;
            }

            if (!instance->Initialize()) {
                std::cerr << instance->Name() << " failed to load" << std::endl;
                exit(-1);
            }
        }

        template<typename E>
        static void InitializeEngineModule(E *& ptr, E * instance, const bool log) {
            InitializeEngineModule<E>(instance, log);
            ptr = instance;
        }
    };

    void Engine::Initialize() {
        // We need to initialize everything on renderer thread
        CHECK_IS_APPLICATION_THREAD();

        if (!IsInitializing()) {
            throw std::runtime_error("Engine::PreInitialize not called");
        }
        std::unique_lock<std::shared_mutex> ul(startupShutdown_);

        STRATUS_LOG << "Engine initializing" << std::endl;

        InitInput_();
        InitEntityManager_();
        InitTaskSystem_();
        InitMaterialManager_();
        InitResourceManager_();
        InitWindow_();
        InitGraphicsDriver_();
        InitRenderer_();

        // Initialize application last
        EngineModuleInit::InitializeEngineModule(Application::Instance(), true);

        STRATUS_LOG << "Initialization complete" << std::endl;
        isInitializing_.store(false);
    }

    void Engine::InitLog_() {
        EngineModuleInit::InitializeEngineModule(Log::Instance_(), new Log(), false);
    }

    void Engine::InitInput_() {
        EngineModuleInit::InitializeEngineModule(InputManager::Instance_(), new InputManager(), true);
    }

    void Engine::InitGraphicsDriver_() {
        GraphicsDriver::Initialize();
    }

    void Engine::InitEntityManager_() {
        EngineModuleInit::InitializeEngineModule(EntityManager::Instance_(), new EntityManager(), true);
    }

    void Engine::InitApplicationThread_() {
        ApplicationThread::Instance_() = new ApplicationThread();
    }

    void Engine::InitTaskSystem_() {
        EngineModuleInit::InitializeEngineModule(TaskSystem::Instance_(), new TaskSystem(), true);
    }

    void Engine::InitMaterialManager_() {
        EngineModuleInit::InitializeEngineModule(MaterialManager::Instance_(), new MaterialManager(), true);
    }

    void Engine::InitResourceManager_() {
        EngineModuleInit::InitializeEngineModule(ResourceManager::Instance_(), new ResourceManager(), true);
    }

    void Engine::InitWindow_() {
        EngineModuleInit::InitializeEngineModule(Window::Instance_(), new Window(1600, 900), true);
    }

    void Engine::InitRenderer_() {
        RendererParams params;
        params.appName = Application::Instance()->GetAppName();
        params.fovy = Degrees(75.0f);
        params.vsyncEnabled = false;

        EngineModuleInit::InitializeEngineModule(RendererFrontend::Instance_(), new RendererFrontend(params), true);
    }

    // Should be called before Shutdown()
    void Engine::BeginShutDown() {
        if (IsShuttingDown()) return;
        isShuttingDown_.store(true);
    }

    // Begins shutdown sequence for engine and all core subsystems
    void Engine::ShutDown() {
        if (!IsShuttingDown()) {
            throw std::runtime_error("Must call Engine::BeginShutdown before Engine::ShutDown");
        }

        STRATUS_LOG << "Engine shutting down" << std::endl;

        // Application should shut down first
        ShutdownResourceAndDelete_(Application::Instance_());
        ShutdownResourceAndDelete_(InputManager::Instance_());
        ShutdownResourceAndDelete_(ResourceManager::Instance_());
        ShutdownResourceAndDelete_(MaterialManager::Instance_());
        ShutdownResourceAndDelete_(RendererFrontend::Instance_());
        ShutdownResourceAndDelete_(Window::Instance_());
        ShutdownResourceAndDelete_(EntityManager::Instance_());
        ShutdownResourceAndDelete_(TaskSystem::Instance_());
        // This one does not have a specialized instance
        GraphicsDriver::Shutdown();
        // This one does not have a shutdown routine
        DeleteResource_(ApplicationThread::Instance_());
        ShutdownResourceAndDelete_(Log::Instance_());
    }

    // Processes the next full system frame, including rendering. Returns false only
    // if the main engine loop should stop.
    SystemStatus Engine::Frame() {
        // Validate
        CHECK_IS_APPLICATION_THREAD();

        if (IsInitializing()) return SystemStatus::SYSTEM_CONTINUE;
        if (IsShuttingDown()) return SystemStatus::SYSTEM_SHUTDOWN;

        //std::shared_lock<std::shared_mutex> sl(_mainLoop);

        // Calculate new frame time
        const auto end = std::chrono::system_clock::now();
        //const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - _stats.prevFrameStart).count();
        const double duration = std::chrono::duration<double, std::milli>(end - stats_.prevFrameStart).count();
        const auto requestedFrameTimingMsec = 1000.0 / double(_params.maxFrameRate);
        // Make sure we haven't exceeded the max frame rate
        //if (frameRate > _params.maxFrameRate) {
        if (duration < requestedFrameTimingMsec) {
            //std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            return SystemStatus::SYSTEM_CONTINUE;
        }


        const double deltaSeconds = duration / 1000.0;
        //const double frameRate = 1.0 / deltaSeconds;

        //sl.unlock();

        //std::unique_lock<std::shared_mutex> ul(_mainLoop);

        // Frame counter should always be +1 for each valid frame
        ++stats_.currentFrame;

        // Update prev frame start to be the beginning of this current frame
        stats_.prevFrameStart = end;

        //ul.unlock();

        SystemStatus status;
        #define UPDATE_MODULE(name)                                     \
            status = name::Instance()->Update(deltaSeconds);            \
            if (status != SystemStatus::SYSTEM_CONTINUE) return status;

        // Update core modules
        UPDATE_MODULE(Log)
        UPDATE_MODULE(InputManager)
        UPDATE_MODULE(EntityManager)
        UPDATE_MODULE(TaskSystem)
        UPDATE_MODULE(MaterialManager)
        UPDATE_MODULE(ResourceManager)
        UPDATE_MODULE(Window)
        UPDATE_MODULE(RendererFrontend)

        // Finish with update to application
        return Application::Instance()->Update(deltaSeconds);

        #undef UPDATE_MODULE
    }

    // Main thread is where both engine + application run
    Thread * Engine::GetMainThread() const {
        return main_;
    }
}