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
        worldLight_ = stratus::InfiniteLightPtr(new stratus::InfiniteLight(true));
        worldLight_->SetRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(10.0f), stratus::Degrees(0.0f)));
        worldLight_->SetColor(lightColor);
        worldLight_->SetAtmosphereColor(atmosphereColor);
        worldLight_->SetIntensity(intensity);
        INSTANCE(RendererFrontend)->SetWorldLight(worldLight_);
    }

    virtual ~WorldLightController() {
        INSTANCE(RendererFrontend)->ClearWorldLight();
    }

    void SetRotation(const stratus::Rotation& r) {
        worldLight_->SetRotation(r);
    }

    void HandleInput(const stratus::MouseState& mouse, const std::vector<SDL_Event>& input, const double deltaSeconds) {
        const double lightRotationSpeed = rotationSpeeds_[rotationIndex_];
        const double lightIncreaseSpeed = 5.0;
        const double minLightBrightness = 0.25;
        const double maxLightBrightness = 30.0;
        const double atmosphericIncreaseSpeed = 0.15;
        const double maxAtomsphericIncreasePerFrame = atmosphericIncreaseSpeed * (1.0 / 60.0);
        double particleDensity = worldLight_->GetAtmosphericParticleDensity();
        double scatterControl = worldLight_->GetAtmosphericScatterControl();
        double lightIntensity = worldLight_->GetIntensity();
        
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
                                worldLightMoveDirection_ = -worldLightMoveDirection_;
                            }
                            break;
                        case SDL_SCANCODE_P:
                            if (released) {
                                STRATUS_LOG << "World Light Movement Toggled" << std::endl;
                                worldLightPaused_ = !worldLightPaused_;
                            }
                            break;
                        case SDL_SCANCODE_E:
                            if (released) {
                                STRATUS_LOG << "World Lighting Toggled" << std::endl;
                                worldLight_->SetEnabled( !worldLight_->GetEnabled() );
                            }
                            break;
                        case SDL_SCANCODE_G: {
                            if (released) {
                                STRATUS_LOG << "Global Illumination Toggled" << std::endl;
                                auto settings = INSTANCE(RendererFrontend)->GetSettings();
                                const bool enabled = settings.globalIlluminationEnabled;
                                settings.globalIlluminationEnabled = !enabled;
                                INSTANCE(RendererFrontend)->SetSettings(settings);
                            }
                            break;
                        }
                        case SDL_SCANCODE_O: {
                            if (released) {
                                rotationIndex_ = (rotationIndex_ + 1) % rotationSpeeds_.size();
                                STRATUS_LOG << "Rotation Speed: " << rotationSpeeds_[rotationIndex_] << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_K: {
                            if (released) {
                                rotationIndex_ = rotationIndex_ == 0 ? rotationSpeeds_.size() - 1 : rotationIndex_ - 1;
                                STRATUS_LOG << "Rotation Speed: " << rotationSpeeds_[rotationIndex_] << std::endl;
                            }
                            break;
                        }
                        case SDL_SCANCODE_MINUS:
                            if (released) {
                                lightIntensity = lightIntensity - lightIncreaseSpeed * deltaSeconds;
                                lightIntensity = std::max(minLightBrightness, std::min(maxLightBrightness, lightIntensity));
                                STRATUS_LOG << "Light Intensity: " << lightIntensity << std::endl;
                                worldLight_->SetIntensity(lightIntensity);
                            }
                            break;
                        case SDL_SCANCODE_EQUALS:
                            if (released) {
                                lightIntensity = lightIntensity + lightIncreaseSpeed * deltaSeconds;
                                lightIntensity = std::max(minLightBrightness, std::min(maxLightBrightness, lightIntensity));
                                STRATUS_LOG << "Light Intensity: " << lightIntensity << std::endl;
                                worldLight_->SetIntensity(lightIntensity);
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

        worldLight_->SetAtmosphericLightingConstants(particleDensity, scatterControl);

        if (!worldLightPaused_) {
            worldLight_->OffsetRotation(glm::vec3(worldLightMoveDirection_ * lightRotationSpeed * deltaSeconds, 0.0f, 0.0f));
            //worldLight_->OffsetRotation(glm::vec3(0.0f, worldLightMoveDirection_ * lightRotationSpeed * deltaSeconds, 0.0f));
            //STRATUS_LOG << worldLight_->GetRotation() << std::endl;
        }

        worldLight_->SetPosition(INSTANCE(RendererFrontend)->GetCamera()->GetPosition());
    }

private:
    std::vector<double> rotationSpeeds_ = std::vector<double>{ 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0 };
    size_t rotationIndex_ = 0;
    float worldLightMoveDirection_ = 1.0; // -1.0 reverses it
    stratus::InfiniteLightPtr worldLight_;
    bool worldLightPaused_ = true;
};