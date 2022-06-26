#include "StratusCommon.h"
#include "glm/glm.hpp"
#include <iostream>
#include <StratusPipeline.h>
#include <StratusCamera.h>
#include <chrono>
#include "StratusEngine.h"
#include "StratusResourceManager.h"
#include "StratusLog.h"
#include "StratusRendererFrontend.h"
#include <StratusLight.h>
#include <StratusUtils.h>
#include <memory>
#include <filesystem>

class RandomLightMover { //: public stratus::Entity {
    glm::vec3 _direction = glm::vec3(0.0f);

    void _changeDirection() {
        float xModifier = rand() % 100 > 50 ? -1.0f : 1.0f;
        float yModifier = 0.0; // rand() % 100 > 50 ? -1.0f : 1.0f;
        float zModifier = rand() % 100 > 50 ? -1.0f : 1.0f;
        _direction.x = (rand() % 100) > 50 ? 1.0f : 0.0f;
        _direction.y = (rand() % 100) > 50 ? 1.0f : 0.0f;
        _direction.z = (rand() % 100) > 50 ? 1.0f : 0.0f;

        _direction = _direction * glm::vec3(xModifier, yModifier, zModifier);
    }

    double _elapsedSec = 0.0;

public:
    stratus::EntityPtr cube;
    stratus::LightPtr light;
    glm::vec3 position;
    glm::vec3 speed;

    RandomLightMover() {
        cube = stratus::ResourceManager::Instance()->CreateCube();
        cube->GetRenderNode()->SetMaterial(stratus::MaterialManager::Instance()->CreateDefault());
        cube->GetRenderNode()->EnableLightInteraction(false);
        //cube->scale = glm::vec3(0.25f, 0.25f, 0.25f);
        cube->SetLocalScale(glm::vec3(1.0f));
        light = stratus::LightPtr(new stratus::PointLight());
        _changeDirection();
    }

    void addToScene() const {
        //stratus::RendererFrontend::Instance()->AddStaticEntity(cube);
        //r.addPointLight(light.get());
        stratus::RendererFrontend::Instance()->AddLight(light);
    }

    void removeFromScene() const {
        //stratus::RendererFrontend::Instance()->RemoveEntity(cube);
        //r.addPointLight(light.get());
        stratus::RendererFrontend::Instance()->RemoveLight(light);
    }

    virtual void update(double deltaSeconds) {
        position = position + speed * _direction * float(deltaSeconds);
        cube->SetLocalPosition(position);
        light->position = position;
        stratus::MaterialPtr m = cube->GetRenderNode()->GetMeshContainer(0)->material;
        m->SetDiffuseColor(light->getColor());

        _elapsedSec += deltaSeconds;
        if (_elapsedSec > 5.0) {
            _elapsedSec = 0.0;
            _changeDirection();
        }
    }
};

struct StationaryLight : public RandomLightMover {
    StationaryLight() : RandomLightMover() {}

    void update(double deltaSeconds) override {
        cube->SetLocalPosition(position);
        light->position = position;
        stratus::MaterialPtr m = cube->GetRenderNode()->GetMeshContainer(0)->material;
        m->SetDiffuseColor(light->getColor());
    }
};

class Forest : public stratus::Application {
public:
    virtual ~Forest() = default;

    const char * GetAppName() const override {
        return "Forest";
    }

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing " << GetAppName() << std::endl;

        camera = stratus::CameraPtr(new stratus::Camera());
        stratus::RendererFrontend::Instance()->SetCamera(camera);
        worldLight = stratus::InfiniteLightPtr(new stratus::InfiniteLight(true));
        //worldLight->setRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(-90.0f), stratus::Degrees(0.0f)));

        stratus::RendererFrontend::Instance()->SetAtmosphericShadowing(0.1f, 0.15f);

        // Disable culling for this model since there are some weird parts that seem to be reversed
        stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../local/southern_railway_stables_weighbridge/scene.gltf", stratus::RenderFaceCulling::CULLING_NONE);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { forest = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(forest); });

        bool running = true;

        cameraLight = stratus::LightPtr(new stratus::PointLight());
        cameraLight->setCastsShadows(false);
        cameraLight->setIntensity(1200.0f);

        if (camLightEnabled) {
            stratus::RendererFrontend::Instance()->AddLight(cameraLight);
        }

        worldLight->setRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(10.0f), stratus::Degrees(0.0f)));

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(const double deltaSeconds) override {
        float value = 1.0f;
        if (stratus::Engine::Instance()->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << " (" << (deltaSeconds * 1000.0) << " ms)" << std::endl;
        }
        const float lightIncreaseSpeed = 5.0f;
        const float maxLightBrightness = 30.0f;

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        // Check for key/mouse events
        for (auto e : stratus::RendererFrontend::Instance()->PollInputEvents()) {
            switch (e.type) {
                case SDL_QUIT:
                    return stratus::SystemStatus::SYSTEM_SHUTDOWN;
                case SDL_MOUSEMOTION:
                    camera->modifyAngle(stratus::Degrees(0.0f), stratus::Degrees(-e.motion.xrel), stratus::Degrees(0.0f));
                    //STRATUS_LOG << camera.getRotation() << std::endl;
                    break;
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    bool released = e.type == SDL_KEYUP;
                    SDL_Scancode key = e.key.keysym.scancode;
                    switch (key) {
                        case SDL_SCANCODE_ESCAPE:
                            if (released) {
                                return stratus::SystemStatus::SYSTEM_SHUTDOWN;
                            }
                            break;
                        case SDL_SCANCODE_SPACE:
                            if (!released) {
                                currentCamSpeed = slowCamSpeed;
                            }
                            else {
                                currentCamSpeed = maxCamSpeed;
                            }
                            break;
                        case SDL_SCANCODE_W:
                        case SDL_SCANCODE_S:
                            if (!released) {
                                cameraSpeed.x = key == SDL_SCANCODE_W ? currentCamSpeed : -currentCamSpeed;
                            } else {
                                cameraSpeed.x = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_A:
                        case SDL_SCANCODE_D:
                            if (!released) {
                                cameraSpeed.y = key == SDL_SCANCODE_D ? currentCamSpeed : -currentCamSpeed;
                            } else {
                                cameraSpeed.y = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_F:
                            if (released) {
                                camLightEnabled = !camLightEnabled;
                                
                                if (camLightEnabled) {
                                    stratus::RendererFrontend::Instance()->AddLight(cameraLight);
                                }
                                else {
                                    stratus::RendererFrontend::Instance()->RemoveLight(cameraLight);
                                }
                            }

                            break;
                        case SDL_SCANCODE_R:
                            if (released) {
                                stratus::RendererFrontend::Instance()->RecompileShaders();
                            }
                            break;
                        case SDL_SCANCODE_I:
                            if (released) {
                                worldLight->setEnabled( !worldLight->getEnabled() );
                            }
                            break;
                        case SDL_SCANCODE_P:
                            if (released) {
                                worldLightPaused = !worldLightPaused;
                            }
                            break;
                        case SDL_SCANCODE_UP: {
                            float brightness = worldLight->getIntensity() + lightIncreaseSpeed * deltaSeconds;
                            brightness = std::min(maxLightBrightness, brightness);
                            worldLight->setIntensity(brightness);
                            STRATUS_LOG << "Brightness: " << brightness << std::endl;
                            break;
                        }
                        case SDL_SCANCODE_DOWN: {
                            float brightness = worldLight->getIntensity() - lightIncreaseSpeed * deltaSeconds;
                            worldLight->setIntensity(brightness);
                            STRATUS_LOG << "Brightness: " << brightness << std::endl;
                            break;
                        }
                        case SDL_SCANCODE_1: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(1200.0f);
                                mover->light->setColor(1.0f, 1.0f, 0.5f);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(1000.0);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_3: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(1500.0f);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_4: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(2000.0f);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_5: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(3000.0f);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_6: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(6000.0f);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_7: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(12000.0f);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_8: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(24000.0f);
                                mover->light->setColor(1.0f, 0.75f, 0.5);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_9: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(48000.0f);
                                mover->light->setColor(1.0f, 0.75f, 0.5);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_0: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(65000.0f);
                                mover->light->setColor(1.0f, 1.0f, 1.0f);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_C: {
                            for (auto& light : lightMovers) {
                                light->removeFromScene();
                            }
                            lightMovers.clear();
                            break;
                        }
                        default: break;
                    }
                    break;
                }
                default: break;
            }
        }

        // worldLight->setRotation(glm::vec3(75.0f, 0.0f, 0.0f));
        //worldLight->setRotation(stratus::Rotation(stratus::Degrees(30.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
        if (!worldLightPaused) {
            worldLight->offsetRotation(glm::vec3(value * deltaSeconds, 0.0f, 0.0f));
        }

        //renderer->toggleWorldLighting(worldLightEnabled);
        stratus::RendererFrontend::Instance()->SetWorldLight(worldLight);
        // worldLight->setColor(glm::vec3(1.0f, 0.75f, 0.5));
        // worldLight->setColor(glm::vec3(1.0f, 0.75f, 0.75f));
        worldLight->setColor(glm::vec3(1.0f));
        worldLight->setPosition(camera->getPosition());
        //worldLight->setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
        //renderer->setWorldLight(worldLight);

        // Check mouse state
        uint32_t buttonState = stratus::RendererFrontend::Instance()->GetMouseState().mask;
        cameraSpeed.z = 0.0f;
        if ((buttonState & SDL_BUTTON_LMASK) != 0) { // left mouse button
            cameraSpeed.z = -currentCamSpeed;
        }
        else if ((buttonState & SDL_BUTTON_RMASK) != 0) { // right mouse button
            cameraSpeed.z = currentCamSpeed;
        }

        camera->setSpeed(cameraSpeed.y, cameraSpeed.z, cameraSpeed.x);
        camera->update(deltaSeconds);

        cameraLight->position = camera->getPosition();
        stratus::RendererFrontend::Instance()->SetClearColor(glm::vec4(90.0f, 0.0f, 0.0f, 1.0f));

        if (forest) {
           forest->SetLocalPosition(glm::vec3(0.0f));
           forest->SetLocalScale(glm::vec3(4.0f));
           auto rot = forest->GetLocalRotation();
           rot.y = stratus::Degrees(-90.0f);
           //forest->SetLocalRotation(rot);
           //forest->SetLocalRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
        }

        //renderer->addDrawable(rocks);

        // Add the camera's light
        //if (camLightEnabled) renderer->addPointLight(&cameraLight);
        //for (auto & entity : entities) {
        //    renderer->addDrawable(entity);
        //}

        for (auto & mover : lightMovers) {
            mover->update(deltaSeconds);
        }
        //renderer->end(camera);

        //// 0 lets it run as fast as it can
        //SDL_GL_SetSwapInterval(0);
        //// Swap front and back buffer
        //SDL_GL_SwapWindow(window);

        return stratus::SystemStatus::SYSTEM_CONTINUE;
    }

    // Perform any resource cleanup
    virtual void Shutdown() override {
    }

private:
    stratus::EntityPtr forest;
    std::vector<stratus::EntityPtr> entities;
    stratus::CameraPtr camera;
    glm::vec3 cameraSpeed;
    const float maxCamSpeed = 50.0f;
    const float slowCamSpeed = 10.0f;
    float currentCamSpeed = maxCamSpeed;
    stratus::LightPtr cameraLight;
    stratus::InfiniteLightPtr worldLight;
    std::vector<std::unique_ptr<RandomLightMover>> lightMovers;
    bool camLightEnabled = false;
    bool worldLightPaused = true;
};

STRATUS_ENTRY_POINT(Forest)