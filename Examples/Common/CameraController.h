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
        _camera = stratus::CameraPtr(new stratus::Camera());
        stratus::RendererFrontend::Instance()->SetCamera(_camera);

        _cameraLight = stratus::LightPtr(new stratus::PointLight(/* staticLight = */ false));
        _cameraLight->setCastsShadows(false);
        _cameraLight->setIntensity(600.0f);

        if (_cameraLightEnabled) {
            stratus::RendererFrontend::Instance()->AddLight(_cameraLight);
        }
    }

    // This class is deleted when the Window is deleted so the Renderer has likely already
    // been taken offline. The check is for if the camera controller is removed while the engine
    // is still running.
    virtual ~CameraController() {
        INSTANCE(RendererFrontend)->SetCamera(nullptr);

        if (_cameraLightEnabled) {
            INSTANCE(RendererFrontend)->RemoveLight(_cameraLight);
        }
    }

    void HandleInput(const stratus::MouseState& mouse, const std::vector<SDL_Event>& input, const double deltaSeconds) {
        const float camSpeed = 100.0f;

        // Handle WASD movement
        for (auto e : input) {
            switch (e.type) {
                case SDL_MOUSEMOTION:
                    if (_cameraRotateEnabled) {
                        _camera->modifyAngle(stratus::Degrees(0.0f), stratus::Degrees(-e.motion.xrel), stratus::Degrees(0.0f));
                    }
                    //STRATUS_LOG << camera.getRotation() << std::endl;
                    break;
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    bool released = e.type == SDL_KEYUP;
                    SDL_Scancode key = e.key.keysym.scancode;
                    switch (key) {
                        case SDL_SCANCODE_SPACE:
                            if (!released && _cameraMoveEnabled) {
                                _camSpeedDivide = 1.0f;
                            }
                            else {
                                _camSpeedDivide = 0.25f;
                            }
                            break;
                        case SDL_SCANCODE_LSHIFT:
                            if (!released && _cameraMoveEnabled) {
                                _camSpeedDivide = 0.125f;
                            }
                            else {
                                _camSpeedDivide = 0.25f;
                            }
                            break;
                        case SDL_SCANCODE_LCTRL:
                            if (!released && _cameraMoveEnabled) {
                                _camSpeedDivide = 0.05f;
                            }
                            else {
                                _camSpeedDivide = 0.25f;
                            }
                            break;
                        case SDL_SCANCODE_W:
                        case SDL_SCANCODE_S:
                            if (!released && _cameraMoveEnabled) {
                                _cameraSpeed.x = key == SDL_SCANCODE_W ? camSpeed : -camSpeed;
                            } else {
                                _cameraSpeed.x = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_A:
                        case SDL_SCANCODE_D:
                            if (!released && _cameraMoveEnabled) {
                                _cameraSpeed.y = key == SDL_SCANCODE_D ? camSpeed : -camSpeed;
                            } else {
                                _cameraSpeed.y = 0.0f;
                            }
                            break;
                        case (SDL_SCANCODE_T): {
                            if (released) {
                                _cameraMoveEnabled = !_cameraMoveEnabled;
                            }
                            break;
                        }
                        case (SDL_SCANCODE_Y): {
                            if (released) {
                                _cameraRotateEnabled = !_cameraRotateEnabled;
                            }
                            break;
                        }
                        // Adds or removes the light following the camera
                        case SDL_SCANCODE_F:
                            if (released) {
                                _cameraLightEnabled = !_cameraLightEnabled;
                                
                                if (_cameraLightEnabled) {
                                    stratus::RendererFrontend::Instance()->AddLight(_cameraLight);
                                }
                                else {
                                    stratus::RendererFrontend::Instance()->RemoveLight(_cameraLight);
                                }
                            }

                            break;
                        case SDL_SCANCODE_V:
                            if (released) {
                                STRATUS_LOG << "Camera Position: " << INSTANCE(RendererFrontend)->GetCamera()->getPosition() << std::endl;
                            }
                            break;
                    }
                }
            }
        }

        // Check mouse state for move up/down
        uint32_t buttonState = mouse.mask;
        _cameraSpeed.z = 0.0f;
        if (_cameraMoveEnabled) {
            if ((buttonState & SDL_BUTTON_LMASK) != 0) { // left mouse button
                _cameraSpeed.z = -camSpeed;
            }
            else if ((buttonState & SDL_BUTTON_RMASK) != 0) { // right mouse button
                _cameraSpeed.z = camSpeed;
            }
        }

        // Final camera speed update
        glm::vec3 tmpCamSpeed = _cameraSpeed * _camSpeedDivide;
        _camera->setSpeed(tmpCamSpeed.y, tmpCamSpeed.z, tmpCamSpeed.x);

        _cameraLight->SetPosition(_camera->getPosition());
    }

private:
    stratus::CameraPtr _camera;
    stratus::LightPtr _cameraLight;
    bool _cameraLightEnabled = true;
    bool _cameraMoveEnabled = true;
    bool _cameraRotateEnabled = true;
    glm::vec3 _cameraSpeed = glm::vec3(0.0f);
    float _camSpeedDivide = 0.25f; // For slowing camera down
};