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
    Engine * Engine::_instance = nullptr;

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
        Application::_Instance() = app;

        // Delete the instance in case it's left over from a previous run
        delete Engine::_instance;
        Engine::_instance = new Engine(params);

        // Pre-initialize to set up things like the ApplicationThread
        Engine::Instance()->PreInitialize();
        ApplicationThread::Instance()->Queue([]() {
            Engine::Instance()->Initialize();
        });
        ApplicationThread::Instance()->_DispatchAndSynchronize();

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
            ApplicationThread::Instance()->_Dispatch();

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
        ApplicationThread::Instance()->_Dispatch();

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

        _InitLog();
        _InitApplicationThread();

        // Pull the main thread
        _main = ApplicationThread::Instance()->_thread.get();
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
        std::unique_lock<std::shared_mutex> ul(_startupShutdown);

        STRATUS_LOG << "Engine initializing" << std::endl;

        _InitInput();
        _InitEntityManager();
        _InitTaskSystem();
        _InitMaterialManager();
        _InitResourceManager();
        _InitWindow();
        _InitGraphicsDriver();
        _InitRenderer();

        // Initialize application last
        EngineModuleInit::InitializeEngineModule(Application::Instance(), true);

        STRATUS_LOG << "Initialization complete" << std::endl;
        _isInitializing.store(false);
    }

    void Engine::_InitLog() {
        EngineModuleInit::InitializeEngineModule(Log::_Instance(), new Log(), false);
    }

    void Engine::_InitInput() {
        EngineModuleInit::InitializeEngineModule(InputManager::_Instance(), new InputManager(), true);
    }

    void Engine::_InitGraphicsDriver() {
        GraphicsDriver::Initialize();
    }

    void Engine::_InitEntityManager() {
        EngineModuleInit::InitializeEngineModule(EntityManager::_Instance(), new EntityManager(), true);
    }

    void Engine::_InitApplicationThread() {
        ApplicationThread::_Instance() = new ApplicationThread();
    }

    void Engine::_InitTaskSystem() {
        EngineModuleInit::InitializeEngineModule(TaskSystem::_Instance(), new TaskSystem(), true);
    }

    void Engine::_InitMaterialManager() {
        EngineModuleInit::InitializeEngineModule(MaterialManager::_Instance(), new MaterialManager(), true);
    }

    void Engine::_InitResourceManager() {
        EngineModuleInit::InitializeEngineModule(ResourceManager::_Instance(), new ResourceManager(), true);
    }

    void Engine::_InitWindow() {
        EngineModuleInit::InitializeEngineModule(Window::_Instance(), new Window(1920, 1080), true);
    }

    void Engine::_InitRenderer() {
        RendererParams params;
        params.appName = Application::Instance()->GetAppName();
        params.fovy = Degrees(90.0f);
        params.vsyncEnabled = false;

        EngineModuleInit::InitializeEngineModule(RendererFrontend::_Instance(), new RendererFrontend(params), true);
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
        _ShutdownResourceAndDelete(Application::_Instance());
        _ShutdownResourceAndDelete(InputManager::_Instance());
        _ShutdownResourceAndDelete(ResourceManager::_Instance());
        _ShutdownResourceAndDelete(MaterialManager::_Instance());
        _ShutdownResourceAndDelete(RendererFrontend::_Instance());
        _ShutdownResourceAndDelete(Window::_Instance());
        _ShutdownResourceAndDelete(EntityManager::_Instance());
        _ShutdownResourceAndDelete(TaskSystem::_Instance());
        // This one does not have a specialized instance
        GraphicsDriver::Shutdown();
        // This one does not have a shutdown routine
        _DeleteResource(ApplicationThread::_Instance());
        _ShutdownResourceAndDelete(Log::_Instance());
    }

    // Processes the next full system frame, including rendering. Returns false only
    // if the main engine loop should stop.
    SystemStatus Engine::Frame() {
        // Validate
        CHECK_IS_APPLICATION_THREAD();

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
        return _main;
    }
}