#pragma once

#include "StratusSystemModule.h"
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>
#include <memory>
#include "StratusCommon.h"

namespace stratus {
    struct MouseState {
        int x;
        int y;
        uint32_t mask;
    };

    struct InputHandler : public std::enable_shared_from_this<InputHandler> {
        virtual ~InputHandler() = default;
        virtual void HandleInput(const MouseState& mouse, const std::vector<SDL_Event>& input, const double deltaSeconds) = 0;
    };

    typedef std::shared_ptr<InputHandler> InputHandlerPtr;

    SYSTEM_MODULE_CLASS(InputManager)
        friend class Window;

        virtual ~InputManager() = default;

        void AddInputHandler(const InputHandlerPtr&);
        void RemoveInputHandler(const InputHandlerPtr&);
        std::vector<SDL_Event> GetInputEventsLastFrame() const;
        // See SDL_GetMouseState
        MouseState GetMouseStateLastFrame() const;

    private:
        bool Initialize() override;
        SystemStatus Update(const double deltaSeconds) override;
        void Shutdown() override;

        // end SystemModule interface

        void _SetInputData(std::vector<SDL_Event>&, const MouseState&);

    private:
        mutable std::shared_mutex _m;
        std::vector<SDL_Event> _inputEvents;
        MouseState _mouse;
        std::unordered_set<InputHandlerPtr> _inputHandlers;
        std::unordered_set<InputHandlerPtr> _inputHandlersToAdd;
        std::unordered_set<InputHandlerPtr> _inputHandlersToRemove;
    };

    SYSTEM_MODULE_CLASS(Window)
    private:
        Window(uint32_t width, uint32_t height);

        bool Initialize() override;
        SystemStatus Update(const double deltaSeconds) override;
        void Shutdown() override;

    public:
        // end SystemModule interface

        // Window dimensions
        std::pair<uint32_t, uint32_t> GetWindowDims() const;
        void SetWindowDims(const uint32_t width, const uint32_t height);
        bool WindowResizedWithinLastFrame() const;

        // Only useful to internal engine code
        void * GetWindowObject() const;

    private:
        mutable std::shared_mutex _m;
        SDL_Window * _window;
        MouseState _mouse;
        uint32_t _width = 0;
        uint32_t _height = 0;
        uint32_t _prevWidth = 0;
        uint32_t _prevHeight = 0;
        bool _resized = false;
    };
}