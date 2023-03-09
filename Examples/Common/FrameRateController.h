#pragma once

#include "StratusCommon.h"
#include "glm/glm.hpp"
#include "StratusWindow.h"
#include "StratusRendererFrontend.h"
#include "StratusLog.h"
#include "StratusCamera.h"
#include "StratusLight.h"

struct FrameRateController : public stratus::InputHandler {
    FrameRateController() {
        INSTANCE(RendererFrontend)->SetVsyncEnabled(true);
        // 1000 fps is just to get the engine out of the way so SDL can control it with vsync
        _frameRates = {1000, 50, 40, 30};
        INSTANCE(Engine)->SetMaxFrameRate(_frameRates[0]);
    }

    // This class is deleted when the Window is deleted so the Renderer has likely already
    // been taken offline. The check is for if the camera controller is removed while the engine
    // is still running.
    virtual ~FrameRateController() {}

    void HandleInput(const stratus::MouseState& mouse, const std::vector<SDL_Event>& input, const double deltaSeconds) {
        const float camSpeed = 100.0f;

        // Handle WASD movement
        for (auto e : input) {
            switch (e.type) {
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    bool released = e.type == SDL_KEYUP;
                    SDL_Scancode key = e.key.keysym.scancode;
                    switch (key) {
                        // Why B? Because F is used for flash light and R is recompile :(
                        case SDL_SCANCODE_B:
                            if (released) {
                                _frameRateIndex = (_frameRateIndex + 1) % _frameRates.size();
                                INSTANCE(Engine)->SetMaxFrameRate(_frameRates[_frameRateIndex]);
                                STRATUS_LOG << "Max Frame Rate Toggled: " << _frameRates[_frameRateIndex] << std::endl;
                                break;
                            }
                    }
                }
            }
        }
    }

private:
    size_t _frameRateIndex = 0;
    std::vector<uint32_t> _frameRates;
};