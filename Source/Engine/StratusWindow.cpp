#include "StratusWindow.h"
#include "StratusLog.h"
#include "StratusApplicationThread.h"
#include "StratusApplication.h"

namespace stratus {
    InputManager::InputManager() {}

    void InputManager::AddInputHandler(const InputHandlerPtr& ptr) {
        auto ul = std::unique_lock<std::shared_mutex>(m_);
        inputHandlersToAdd_.insert(ptr);
    }

    void InputManager::RemoveInputHandler(const InputHandlerPtr& ptr) {
        auto ul = std::unique_lock<std::shared_mutex>(m_);
        inputHandlersToRemove_.insert(ptr);
    }

    std::vector<SDL_Event> InputManager::GetInputEventsLastFrame() const {
        auto sl = std::shared_lock<std::shared_mutex>(m_);
        return inputEvents_;
    }

    MouseState InputManager::GetMouseStateLastFrame() const {
        auto sl = std::shared_lock<std::shared_mutex>(m_);
        return mouse_;
    }

    bool InputManager::Initialize() { return true; }

    SystemStatus InputManager::Update(const double deltaSeconds) {
        // Commit input handler changes
        inputHandlers_.insert(inputHandlersToAdd_.begin(), inputHandlersToAdd_.end());
        inputHandlers_.erase(inputHandlersToRemove_.begin(), inputHandlersToRemove_.end());
        inputHandlersToAdd_.clear();
        inputHandlersToRemove_.clear();

        // Allow all input handlers to update
        for (const InputHandlerPtr& ptr : inputHandlers_) {
            ptr->HandleInput(mouse_, inputEvents_, deltaSeconds);
        }

        return SystemStatus::SYSTEM_CONTINUE;
    }

    void InputManager::Shutdown() {
        inputHandlers_.clear();
        inputHandlersToAdd_.clear();
        inputHandlersToRemove_.clear();
    }

    void InputManager::SetInputData_(std::vector<SDL_Event>& events, const MouseState& mouse) {
        auto ul = std::unique_lock<std::shared_mutex>(m_);
        inputEvents_ = std::move(events);
        mouse_ = mouse;
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
        window_ = SDL_CreateWindow(Application::Instance()->GetAppName(),
                100, 100, // location x/y on screen
                width_, height_, // width/height of window
                SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL );
        if (window_ == nullptr) {
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
        resized_ = false;
        if (width_ != prevWidth_ || height_ != prevHeight_) {
            resized_ = true;
            SDL_SetWindowSize(window_, width_, height_);
        }
        prevWidth_ = width_;
        prevHeight_ = height_;

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
        mouse_.mask = SDL_GetMouseState(&mouse_.x, &mouse_.y);

        // Tell InputManager about state change
        InputManager::Instance()->SetInputData_(inputEvents, mouse_);

        return SystemStatus::SYSTEM_CONTINUE;
    }

    void Window::Shutdown() {
        // We need this to run on Application thread since it involves the
        // graphics backend to some extent
        CHECK_IS_APPLICATION_THREAD();

        if (window_) {
            SDL_DestroyWindow(window_);
            window_ = nullptr;
            SDL_Quit();
        }
    }

    std::pair<uint32_t, uint32_t> Window::GetWindowDims() const {
        auto sl = std::shared_lock<std::shared_mutex>(m_);
        return std::make_pair(width_, height_);
    }

    void Window::SetWindowDims(const uint32_t width, const uint32_t height) {
        assert(width > 0 && height > 0);
        auto ul = std::unique_lock<std::shared_mutex>(m_);
        width_ = width;
        height_ = height;
    }

    bool Window::WindowResizedWithinLastFrame() const {
        auto sl = std::shared_lock<std::shared_mutex>(m_);
        return resized_;
    }

    void * Window::GetWindowObject() const {
        auto sl = std::shared_lock<std::shared_mutex>(m_);
        return (void *)window_;
    }
}