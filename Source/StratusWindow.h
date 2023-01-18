#pragma once

#include "StratusSystemModule.h"
#include <unordered_map>
#include <shared_mutex>
#include "StratusCommon.h"

namespace stratus {
    struct MouseState {
        int x;
        int y;
        uint32_t mask;
    };

    class Window : public SystemModule {
        friend class Engine;
        friend class EngineModuleInit;

        Window(uint32_t width, uint32_t height);

        bool Initialize() override;
        SystemStatus Update(const double deltaSeconds) override;
        void Shutdown() override;

    public:
        static Window * Instance() { return _instance; }

        // begin SystemModule interface
        const char * Name() const override {
            return "Window";
        }

        // end SystemModule interface

        // Window dimensions
        std::pair<uint32_t, uint32_t> GetWindowDims() const;
        void SetWindowDims(const uint32_t width, const uint32_t height);
        bool WindowResizedWithinLastFrame() const;

        // Input events
        std::vector<SDL_Event> GetInputEventsLastFrame() const;
        // See SDL_GetMouseState
        MouseState GetMouseStateLastFrame() const;

        // Only useful to internal engine code
        void * GetWindowObject() const;

    private:
        static Window * _instance;

        mutable std::shared_mutex _m;
        SDL_Window * _window;
        std::vector<SDL_Event> _inputEvents;
        uint32_t _width = 0;
        uint32_t _height = 0;
        uint32_t _prevWidth = 0;
        uint32_t _prevHeight = 0;
        MouseState _mouse;
        bool _resized = false;
    };
}