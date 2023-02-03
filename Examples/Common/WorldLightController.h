#pragma once

#include "StratusCommon.h"
#include "glm/glm.hpp"
#include "StratusWindow.h"
#include "StratusRendererFrontend.h"
#include "StratusLog.h"
#include "StratusCamera.h"
#include "StratusLight.h"
#include "StratusEngine.h"
#include "StratusResourceManager.h"
#include "StratusUtils.h"

struct WorldLightController : public stratus::InputHandler {
    WorldLightController(const glm::vec3& lightColor) {
        _worldLight = stratus::InfiniteLightPtr(new stratus::InfiniteLight(true));
        _worldLight->setRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(10.0f), stratus::Degrees(0.0f)));
        _worldLight->setColor(lightColor);
        INSTANCE(RendererFrontend)->SetWorldLight(_worldLight);
    }

    virtual ~WorldLightController() {
        INSTANCE(RendererFrontend)->ClearWorldLight();
    }

    void HandleInput(const stratus::MouseState& mouse, const std::vector<SDL_Event>& input, const double deltaSeconds) {
        const float lightRotationSpeed = 3.0f;
        const float lightIncreaseSpeed = 5.0f;
        const float minLightBrightness = 0.25f;
        const float maxLightBrightness = 30.0f;
        const float atmosphericIncreaseSpeed = 1.0f;
        float fogDensity = INSTANCE(RendererFrontend)->GetAtmosphericFogDensity();
        float scatterControl = INSTANCE(RendererFrontend)->GetAtmosphericScatterControl();
        float lightIntensity = _worldLight->getIntensity();
        
        for (auto e : input) {
            switch (e.type) {
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    bool released = e.type == SDL_KEYUP;
                    SDL_Scancode key = e.key.keysym.scancode;
                    switch (key) {
                        case SDL_SCANCODE_I:
                            if (released) {
                                STRATUS_LOG << "World Light Direction Reversed" << std::endl;
                                _worldLightMoveDirection = -_worldLightMoveDirection;
                            }
                            break;
                        case SDL_SCANCODE_P:
                            if (released) {
                                STRATUS_LOG << "World Light Movement Toggled" << std::endl;
                                _worldLightPaused = !_worldLightPaused;
                            }
                            break;
                        case SDL_SCANCODE_E:
                            if (released) {
                                STRATUS_LOG << "World Lighting Toggled" << std::endl;
                                _worldLight->setEnabled( !_worldLight->getEnabled() );
                            }
                            break;
                        case SDL_SCANCODE_G: {
                            if (released) {
                                STRATUS_LOG << "Global Illumination Toggled" << std::endl;
                                const bool enabled = INSTANCE(RendererFrontend)->GetGlobalIlluminationEnabled();
                                INSTANCE(RendererFrontend)->SetGlobalIlluminationEnabled( !enabled );
                            }
                            break;
                        }
                        case SDL_SCANCODE_MINUS:
                            if (released) {
                                lightIntensity = lightIntensity - lightIncreaseSpeed * deltaSeconds;
                                lightIntensity = std::max(minLightBrightness, std::min(maxLightBrightness, lightIntensity));
                                STRATUS_LOG << "Light Intensity: " << lightIntensity << std::endl;
                                _worldLight->setIntensity(lightIntensity);
                            }
                            break;
                        case SDL_SCANCODE_EQUALS:
                            if (released) {
                                lightIntensity = lightIntensity + lightIncreaseSpeed * deltaSeconds;
                                lightIntensity = std::max(minLightBrightness, std::min(maxLightBrightness, lightIntensity));
                                STRATUS_LOG << "Light Intensity: " << lightIntensity << std::endl;
                                _worldLight->setIntensity(lightIntensity);
                            }
                            break;
                        case SDL_SCANCODE_UP: {
                            if (released) {
                                scatterControl = scatterControl + atmosphericIncreaseSpeed * deltaSeconds;
                                STRATUS_LOG << "Scatter Control: " << scatterControl << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_DOWN: {
                            if (released) {
                                scatterControl = scatterControl - atmosphericIncreaseSpeed * deltaSeconds;
                                STRATUS_LOG << "Scatter Control: " << scatterControl << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_LEFT: {
                            if (released) {
                                fogDensity = fogDensity - atmosphericIncreaseSpeed * deltaSeconds;
                                STRATUS_LOG << "Fog Density: " << fogDensity << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_RIGHT: {
                            if (released) {
                                fogDensity = fogDensity + atmosphericIncreaseSpeed * deltaSeconds;
                                STRATUS_LOG << "Fog Density: " << fogDensity << std::endl;
                            }
                            break;
                        }
                    }
                }
            }
        }

        if (!_worldLightPaused) {
            INSTANCE(RendererFrontend)->SetAtmosphericShadowing(fogDensity, scatterControl);
            _worldLight->offsetRotation(glm::vec3(_worldLightMoveDirection * lightRotationSpeed * deltaSeconds, 0.0f, 0.0f));
        }

        _worldLight->setPosition(INSTANCE(RendererFrontend)->GetCamera()->getPosition());
    }

private:
    float _worldLightMoveDirection = 1.0; // -1.0 reverses it
    stratus::InfiniteLightPtr _worldLight;
    bool _worldLightPaused = true;
};