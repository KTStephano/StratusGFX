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
#include <algorithm>
#include <cmath>

struct WorldLightController : public stratus::InputHandler {
    WorldLightController(const glm::vec3& lightColor, const glm::vec3& atmosphereColor, const float intensity = 4.0f) {
        _worldLight = stratus::InfiniteLightPtr(new stratus::InfiniteLight(true));
        _worldLight->SetRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(10.0f), stratus::Degrees(0.0f)));
        _worldLight->SetColor(lightColor);
        _worldLight->SetAtmosphereColor(atmosphereColor);
        _worldLight->SetIntensity(intensity);
        INSTANCE(RendererFrontend)->SetWorldLight(_worldLight);
    }

    virtual ~WorldLightController() {
        INSTANCE(RendererFrontend)->ClearWorldLight();
    }

    void SetRotation(const stratus::Rotation& r) {
        _worldLight->SetRotation(r);
    }

    void HandleInput(const stratus::MouseState& mouse, const std::vector<SDL_Event>& input, const double deltaSeconds) {
        const double lightRotationSpeed = _rotationSpeeds[_rotationIndex];
        const double lightIncreaseSpeed = 5.0;
        const double minLightBrightness = 0.25;
        const double maxLightBrightness = 30.0;
        const double atmosphericIncreaseSpeed = 0.15;
        const double maxAtomsphericIncreasePerFrame = atmosphericIncreaseSpeed * (1.0 / 60.0);
        double particleDensity = _worldLight->GetAtmosphericParticleDensity();
        double scatterControl = _worldLight->GetAtmosphericScatterControl();
        double lightIntensity = _worldLight->GetIntensity();
        
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
                                _worldLight->SetEnabled( !_worldLight->GetEnabled() );
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
                        case SDL_SCANCODE_O: {
                            if (released) {
                                _rotationIndex = (_rotationIndex + 1) % _rotationSpeeds.size();
                                STRATUS_LOG << "Rotation Speed: " << _rotationSpeeds[_rotationIndex] << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_MINUS:
                            if (released) {
                                lightIntensity = lightIntensity - lightIncreaseSpeed * deltaSeconds;
                                lightIntensity = std::max(minLightBrightness, std::min(maxLightBrightness, lightIntensity));
                                STRATUS_LOG << "Light Intensity: " << lightIntensity << std::endl;
                                _worldLight->SetIntensity(lightIntensity);
                            }
                            break;
                        case SDL_SCANCODE_EQUALS:
                            if (released) {
                                lightIntensity = lightIntensity + lightIncreaseSpeed * deltaSeconds;
                                lightIntensity = std::max(minLightBrightness, std::min(maxLightBrightness, lightIntensity));
                                STRATUS_LOG << "Light Intensity: " << lightIntensity << std::endl;
                                _worldLight->SetIntensity(lightIntensity);
                            }
                            break;
                        case SDL_SCANCODE_UP: {
                            if (released) {
                                scatterControl = scatterControl + std::min(atmosphericIncreaseSpeed * deltaSeconds, maxAtomsphericIncreasePerFrame);
                                STRATUS_LOG << "Scatter Control: " << scatterControl << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_DOWN: {
                            if (released) {
                                scatterControl = scatterControl - std::min(atmosphericIncreaseSpeed * deltaSeconds, maxAtomsphericIncreasePerFrame);
                                STRATUS_LOG << "Scatter Control: " << scatterControl << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_LEFT: {
                            if (released) {
                                particleDensity = particleDensity - std::min(atmosphericIncreaseSpeed * deltaSeconds, maxAtomsphericIncreasePerFrame);
                                STRATUS_LOG << "Fog Density: " << particleDensity << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_RIGHT: {
                            if (released) {
                                particleDensity = particleDensity + std::min(atmosphericIncreaseSpeed * deltaSeconds, maxAtomsphericIncreasePerFrame);
                                STRATUS_LOG << "Fog Density: " << particleDensity << std::endl;
                            }
                            break;
                        }
                    }
                }
            }
        }

        _worldLight->SetAtmosphericLightingConstants(particleDensity, scatterControl);

        if (!_worldLightPaused) {
            _worldLight->OffsetRotation(glm::vec3(_worldLightMoveDirection * lightRotationSpeed * deltaSeconds, 0.0f, 0.0f));
        }

        _worldLight->SetPosition(INSTANCE(RendererFrontend)->GetCamera()->GetPosition());
    }

private:
    std::vector<double> _rotationSpeeds = std::vector<double>{ 0.5, 1.0, 2.0, 3.0, 4.0, 5.0 };
    size_t _rotationIndex = 0;
    float _worldLightMoveDirection = 1.0; // -1.0 reverses it
    stratus::InfiniteLightPtr _worldLight;
    bool _worldLightPaused = true;
};