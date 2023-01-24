#include "StratusWindow.h"
#include "StratusLog.h"
#include "StratusApplicationThread.h"
#include "StratusApplication.h"

namespace stratus {
    InputManager::InputManager() {}

    void InputManager::AddInputHandler(const InputHandlerPtr& ptr) {
        auto ul = std::unique_lock<std::shared_mutex>(_m);
        _inputHandlersToAdd.insert(ptr);
    }

    void InputManager::RemoveInputHandler(const InputHandlerPtr& ptr) {
        auto ul = std::unique_lock<std::shared_mutex>(_m);
        _inputHandlersToRemove.insert(ptr);
    }

    std::vector<SDL_Event> InputManager::GetInputEventsLastFrame() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _inputEvents;
    }

    MouseState InputManager::GetMouseStateLastFrame() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _mouse;
    }

    bool InputManager::Initialize() { return true; }

    SystemStatus InputManager::Update(const double deltaSeconds) {
        // Commit input handler changes
        _inputHandlers.insert(_inputHandlersToAdd.begin(), _inputHandlersToAdd.end());
        _inputHandlers.erase(_inputHandlersToRemove.begin(), _inputHandlersToRemove.end());
        _inputHandlersToAdd.clear();
        _inputHandlersToRemove.clear();

        // Allow all input handlers to update
        for (const InputHandlerPtr& ptr : _inputHandlers) {
            ptr->HandleInput(_mouse, _inputEvents, deltaSeconds);
        }

        return SystemStatus::SYSTEM_CONTINUE;
    }

    void InputManager::Shutdown() {
        _inputHandlers.clear();
        _inputHandlersToAdd.clear();
        _inputHandlersToRemove.clear();
    }

    void InputManager::_SetInputData(std::vector<SDL_Event>& events, const MouseState& mouse) {
        auto ul = std::unique_lock<std::shared_mutex>(_m);
        _inputEvents = std::move(events);
        _mouse = mouse;
    }

    Window::Window() : Window(1920, 1080) {}
    
    Window::Window(uint32_t width, uint32_t height) {
        SetWindowDims(width, height);
    }

    bool Window::Initialize() {
        // We need this to run on Application thread since it involves the
        // graphics backend to some extent
        CHECK_IS_APPLICATION_THREAD();

        STRATUS_LOG << "Initializing SDL video" << std::endl;
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            STRATUS_ERROR << "Unable to initialize sdl2" << std::endl;
            STRATUS_ERROR << SDL_GetError() << std::endl;
            return false;
        }

        STRATUS_LOG << "Initializing SDL window" << std::endl;
        _window = SDL_CreateWindow(Application::Instance()->GetAppName(),
                100, 100, // location x/y on screen
                _width, _height, // width/height of window
                SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL );
        if (_window == nullptr) {
            STRATUS_ERROR << "Failed to create sdl window" << std::endl;
            STRATUS_ERROR << SDL_GetError() << std::endl;
            SDL_Quit();
            return false;
        }

        return true;
    }

    SystemStatus Window::Update(const double deltaSeconds) {
        // We need this to run on Application thread since it involves the
        // graphics backend to some extent
        CHECK_IS_APPLICATION_THREAD();

        // Check for window dims change
        _resized = false;
        if (_width != _prevWidth || _height != _prevHeight) {
            _resized = true;
        }
        _prevWidth = _width;
        _prevHeight = _height;

        // Collect window input events
        std::vector<SDL_Event> inputEvents;
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                return SystemStatus::SYSTEM_SHUTDOWN;
            }

            // Filter out everything except for certain input events
            if (e.type == SDL_KEYDOWN ||
                e.type == SDL_KEYUP ||
                e.type == SDL_MOUSEMOTION ||
                e.type == SDL_MOUSEBUTTONDOWN ||
                e.type == SDL_MOUSEBUTTONUP ||
                e.type == SDL_MOUSEWHEEL) {
                
                inputEvents.push_back(e);
            }
        }

        // Check mouse
        _mouse.mask = SDL_GetMouseState(&_mouse.x, &_mouse.y);

        // Tell InputManager about state change
        InputManager::Instance()->_SetInputData(inputEvents, _mouse);

        return SystemStatus::SYSTEM_CONTINUE;
    }

    void Window::Shutdown() {
        // We need this to run on Application thread since it involves the
        // graphics backend to some extent
        CHECK_IS_APPLICATION_THREAD();

        if (_window) {
            SDL_DestroyWindow(_window);
            _window = nullptr;
            SDL_Quit();
        }
    }

    std::pair<uint32_t, uint32_t> Window::GetWindowDims() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return std::make_pair(_width, _height);
    }

    void Window::SetWindowDims(const uint32_t width, const uint32_t height) {
        assert(width > 0 && height > 0);
        auto ul = std::unique_lock<std::shared_mutex>(_m);
        _width = width;
        _height = height;
    }

    bool Window::WindowResizedWithinLastFrame() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return _resized;
    }

    void * Window::GetWindowObject() const {
        auto sl = std::shared_lock<std::shared_mutex>(_m);
        return (void *)_window;
    }
}