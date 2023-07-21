#pragma once

#include "StratusCommon.h"
#include "glm/glm.hpp"
#include "StratusWindow.h"
#include "StratusRendererFrontend.h"
#include "StratusLog.h"
#include "StratusCamera.h"
#include "StratusLight.h"

struct CameraController : public stratus::InputHandler {
    CameraController() {
        camera_ = stratus::CameraPtr(new stratus::Camera(true, true));
        stratus::RendererFrontend::Instance()->SetCamera(camera_);

        cameraLight_ = stratus::LightPtr(new stratus::PointLight(/* staticLight = */ false));
        cameraLight_->SetCastsShadows(false);
        cameraLight_->SetIntensity(600.0f);

        if (cameraLightEnabled_) {
            stratus::RendererFrontend::Instance()->AddLight(cameraLight_);
        }
    }

    // This class is deleted when the Window is deleted so the Renderer has likely already
    // been taken offline. The check is for if the camera controller is removed while the engine
    // is still running.
    virtual ~CameraController() {
        INSTANCE(RendererFrontend)->SetCamera(nullptr);

        if (cameraLightEnabled_) {
            INSTANCE(RendererFrontend)->RemoveLight(cameraLight_);
        }
    }

    void HandleInput(const stratus::MouseState& mouse, const std::vector<SDL_Event>& input, const double deltaSeconds) {
        const float camSpeed = 100.0f;

        // Handle WASD movement
        for (auto e : input) {
            switch (e.type) {
                case SDL_MOUSEMOTION: {
                    const float pitch = cameraLookUpDownEnabled_ ? e.motion.yrel : 0.0f;
                    const float yaw = cameraRotateEnabled_ ? -e.motion.xrel : 0.0f;
                    camera_->ModifyAngle(stratus::Degrees(pitch), stratus::Degrees(yaw), stratus::Degrees(0.0f));
                    //STRATUS_LOG << camera.getRotation() << std::endl;
                    break;
                }
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    bool released = e.type == SDL_KEYUP;
                    SDL_Scancode key = e.key.keysym.scancode;
                    switch (key) {
                        case SDL_SCANCODE_SPACE:
                            if (!released && cameraMoveEnabled_) {
                                camSpeedDivide_ = 1.0f;
                            }
                            else {
                                camSpeedDivide_ = 0.25f;
                            }
                            break;
                        case SDL_SCANCODE_LSHIFT:
                            if (!released && cameraMoveEnabled_) {
                                camSpeedDivide_ = 0.125f;
                            }
                            else {
                                camSpeedDivide_ = 0.25f;
                            }
                            break;
                        case SDL_SCANCODE_LCTRL:
                            if (!released && cameraMoveEnabled_) {
                                camSpeedDivide_ = 0.025f;
                            }
                            else {
                                camSpeedDivide_ = 0.25f;
                            }
                            break;
                        case SDL_SCANCODE_W:
                        case SDL_SCANCODE_S:
                            if (!released && cameraMoveEnabled_) {
                                cameraSpeed_.x = key == SDL_SCANCODE_W ? camSpeed : -camSpeed;
                            } else {
                                cameraSpeed_.x = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_A:
                        case SDL_SCANCODE_D:
                            if (!released && cameraMoveEnabled_) {
                                cameraSpeed_.y = key == SDL_SCANCODE_D ? camSpeed : -camSpeed;
                            } else {
                                cameraSpeed_.y = 0.0f;
                            }
                            break;
                        case (SDL_SCANCODE_T): {
                            if (released) {
                                cameraMoveEnabled_ = !cameraMoveEnabled_;
                            }
                            break;
                        }
                        case (SDL_SCANCODE_Y): {
                            if (released) {
                                cameraRotateEnabled_ = !cameraRotateEnabled_;
                            }
                            break;
                        }
                        case (SDL_SCANCODE_U): {
                            if (released) {
                                cameraLookUpDownEnabled_ = !cameraLookUpDownEnabled_;
                            }
                            break;
                        }
                        // Adds or removes the light following the camera
                        case SDL_SCANCODE_F:
                            if (released) {
                                cameraLightEnabled_ = !cameraLightEnabled_;
                                
                                if (cameraLightEnabled_) {
                                    stratus::RendererFrontend::Instance()->AddLight(cameraLight_);
                                }
                                else {
                                    stratus::RendererFrontend::Instance()->RemoveLight(cameraLight_);
                                }
                            }

                            break;
                        case SDL_SCANCODE_V:
                            if (released) {
                                STRATUS_LOG << "Camera Position: " << INSTANCE(RendererFrontend)->GetCamera()->GetPosition() << std::endl;
                                STRATUS_LOG << "Camera Rotation: " << INSTANCE(RendererFrontend)->GetCamera()->GetRotation() << std::endl;
                            }
                            break;
                        case SDL_SCANCODE_HOME:
                            pitchYawSpeed_.x = -5.0;
                            if (released) {
                                pitchYawSpeed_.x = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_END:
                            pitchYawSpeed_.x = 5.0;
                            if (released) {
                                pitchYawSpeed_.x = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_DELETE:
                            pitchYawSpeed_.y = 5.0;
                            if (released) {
                                pitchYawSpeed_.y = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_PAGEDOWN:
                            pitchYawSpeed_.y = -5.0;
                            if (released) {
                                pitchYawSpeed_.y = 0.0f;
                            } 
                            break;
                    }
                }
            }
        }

        // Check mouse state for move up/down
        uint32_t buttonState = mouse.mask;
        cameraSpeed_.z = 0.0f;
        if (cameraMoveEnabled_) {
            if ((buttonState & SDL_BUTTON_LMASK) != 0) { // left mouse button
                cameraSpeed_.z = -camSpeed;
            }
            else if ((buttonState & SDL_BUTTON_RMASK) != 0) { // right mouse button
                cameraSpeed_.z = camSpeed;
            }
        }

        // Final camera speed update
        glm::vec3 tmpCamSpeed = cameraSpeed_ * camSpeedDivide_;
        camera_->SetSpeed(tmpCamSpeed.y, tmpCamSpeed.z, tmpCamSpeed.x);
        camera_->ModifyAngle(stratus::Degrees(pitchYawSpeed_.x * deltaSeconds), stratus::Degrees(pitchYawSpeed_.y * deltaSeconds), stratus::Degrees(0.0f));

        cameraLight_->SetPosition(camera_->GetPosition());
    }

private:
    stratus::CameraPtr camera_;
    stratus::LightPtr cameraLight_;
    bool cameraLightEnabled_ = false;
    bool cameraMoveEnabled_ = true;
    bool cameraRotateEnabled_ = true;
    bool cameraLookUpDownEnabled_ = false;
    glm::vec3 cameraSpeed_ = glm::vec3(0.0f);
    glm::vec2 pitchYawSpeed_ = glm::vec2(0.0f);
    float camSpeedDivide_ = 0.25f; // For slowing camera down
};