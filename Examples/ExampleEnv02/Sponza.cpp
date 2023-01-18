#include "StratusCommon.h"
#include "glm/glm.hpp"
#include <iostream>
#include <StratusPipeline.h>
#include <StratusCamera.h>
#include "StratusAsync.h"
#include <chrono>
#include "StratusEngine.h"
#include "StratusResourceManager.h"
#include "StratusLog.h"
#include "StratusRendererFrontend.h"
#include "StratusWindow.h"
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
    bool spawnPhysicalMarker;

    RandomLightMover(const bool spawnPhysicalMarker = true, stratus::LightPtr ptr = nullptr) 
        : spawnPhysicalMarker(spawnPhysicalMarker) {
        cube = stratus::ResourceManager::Instance()->CreateCube();
        cube->GetRenderNode()->SetMaterial(stratus::MaterialManager::Instance()->CreateDefault());
        cube->GetRenderNode()->EnableLightInteraction(false);
        //cube->scale = glm::vec3(0.25f, 0.25f, 0.25f);
        cube->SetLocalScale(glm::vec3(1.0f));
        light = ptr ? ptr : stratus::LightPtr(new stratus::PointLight());
        _changeDirection();
    }

    void addToScene() const {
        if (spawnPhysicalMarker) {
            stratus::RendererFrontend::Instance()->AddStaticEntity(cube);
        }
        //r.addPointLight(light.get());
        stratus::RendererFrontend::Instance()->AddLight(light);
    }

    void removeFromScene() const {
        stratus::RendererFrontend::Instance()->RemoveEntity(cube);
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
    StationaryLight(const bool spawnPhysicalMarker = true) 
        : RandomLightMover(spawnPhysicalMarker) {}

    void update(double deltaSeconds) override {
        cube->SetLocalPosition(position);
        light->position = position;
        stratus::MaterialPtr m = cube->GetRenderNode()->GetMeshContainer(0)->material;
        m->SetDiffuseColor(light->getColor());
    }
};

struct FakeRTGILight : public RandomLightMover {
    FakeRTGILight(const bool spawnPhysicalMarker = true)
        : RandomLightMover(spawnPhysicalMarker, stratus::LightPtr(new stratus::VirtualPointLight())) {}

    void update(double deltaSeconds) override {
        cube->SetLocalPosition(position);
        light->position = position;
        stratus::MaterialPtr m = cube->GetRenderNode()->GetMeshContainer(0)->material;
        m->SetDiffuseColor(light->getColor());
    }
};

class Sponza : public stratus::Application {
public:
    virtual ~Sponza() = default;

    const char * GetAppName() const override {
        return "Sponza";
    }

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing " << GetAppName() << std::endl;

        camera = stratus::CameraPtr(new stratus::Camera());
        stratus::RendererFrontend::Instance()->SetCamera(camera);
        worldLight = stratus::InfiniteLightPtr(new stratus::InfiniteLight(true));

        // Moonlight
        //worldLight->setColor(glm::vec3(80.0f / 255.0f, 104.0f / 255.0f, 134.0f / 255.0f));
        //worldLight->setIntensity(0.5f);

        //stratus::RendererFrontend::Instance()->SetAtmosphericShadowing(0.2f, 0.3f);

        // Disable culling for this model since there are some weird parts that seem to be reversed
        stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../local/sponza_scene/scene.gltf", stratus::RenderFaceCulling::CULLING_NONE);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { sponza = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(sponza); });

        stratus::RendererFrontend::Instance()->SetSkybox(stratus::ResourceManager::Instance()->LoadCubeMap("../resources/textures/Skyboxes/learnopengl/sbox_", false, "jpg"));

        bool running = true;

        cameraLight = stratus::LightPtr(new stratus::PointLight());
        cameraLight->setCastsShadows(false);
        cameraLight->setIntensity(1200.0f);

        if (camLightEnabled) {
            stratus::RendererFrontend::Instance()->AddLight(cameraLight);
        }

        worldLight->setRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(10.0f), stratus::Degrees(0.0f)));

        // for (int i = 0; i < 64; ++i) {
        //     float x = rand() % 600;
        //     float y = rand() % 600;
        //     float z = rand() % 200;
        //     stratus::VirtualPointLight * vpl = new stratus::VirtualPointLight();
        //     vpl->setIntensity(worldLight->getIntensity() * 50.0f);
        //     vpl->position = glm::vec3(x, y, z);
        //     vpl->setColor(worldLight->getColor());
        //     World()->AddLight(stratus::LightPtr((stratus::Light *)vpl));
        // }

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(const double deltaSeconds) override {
        float value = 3.0f;
        if (stratus::Engine::Instance()->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << " (" << (deltaSeconds * 1000.0) << " ms)" << std::endl;
        }

        const float camSpeed = 100.0f;
        const float lightIncreaseSpeed = 5.0f;
        const float maxLightBrightness = 30.0f;

        const float atmosphericIncreaseSpeed = 1.0f;
        float fogDensity = stratus::RendererFrontend::Instance()->GetAtmosphericFogDensity();
        float scatterControl = stratus::RendererFrontend::Instance()->GetAtmosphericScatterControl();

        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        // Check for key/mouse events
        auto events = Window()->GetInputEventsLastFrame();
        for (auto e : events) {
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
                                camSpeedDivide = 1.0f;
                            }
                            else {
                                camSpeedDivide = 0.25f;
                            }
                            break;
                        case SDL_SCANCODE_W:
                        case SDL_SCANCODE_S:
                            if (!released) {
                                cameraSpeed.x = key == SDL_SCANCODE_W ? camSpeed : -camSpeed;
                            } else {
                                cameraSpeed.x = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_A:
                        case SDL_SCANCODE_D:
                            if (!released) {
                                cameraSpeed.y = key == SDL_SCANCODE_D ? camSpeed : -camSpeed;
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
                                worldLightMoveDirection = -worldLightMoveDirection;
                            }
                            break;
                        case SDL_SCANCODE_L:
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
                            scatterControl = scatterControl + atmosphericIncreaseSpeed * deltaSeconds;
                            STRATUS_LOG << "Scatter Control: " << scatterControl << std::endl;
                            break;
                        }
                        case SDL_SCANCODE_DOWN: {
                            scatterControl = scatterControl - atmosphericIncreaseSpeed * deltaSeconds;
                            STRATUS_LOG << "Scatter Control: " << scatterControl << std::endl;
                            break;
                        }
                        case SDL_SCANCODE_LEFT: {
                            fogDensity = fogDensity - atmosphericIncreaseSpeed * deltaSeconds;
                            STRATUS_LOG << "Fog Density: " << fogDensity << std::endl;
                            break;
                        }
                        case SDL_SCANCODE_RIGHT: {
                            fogDensity = fogDensity + atmosphericIncreaseSpeed * deltaSeconds;
                            STRATUS_LOG << "Fog Density: " << fogDensity << std::endl;
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
                                std::unique_ptr<RandomLightMover> mover(new FakeRTGILight(/*spawnPhysicalMarker = */ false));
                                mover->light->setIntensity(worldLight->getIntensity() * 100);
                                mover->light->setColor(warmMorningColor);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_3: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new FakeRTGILight(/*spawnPhysicalMarker = */ false));
                                mover->light->setIntensity(worldLight->getIntensity() * 50);
                                mover->light->setColor(warmMorningColor);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_4: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new FakeRTGILight(/*spawnPhysicalMarker = */ false));
                                mover->light->setIntensity(worldLight->getIntensity() * 10);
                                const auto worldLightColor = glm::vec3(1.0f, 0.0f, 0.0f);
                                mover->light->setColor(worldLightColor.r, worldLightColor.g, worldLightColor.b);
                                ((stratus::VirtualPointLight *)mover->light.get())->SetNumShadowSamples(9);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_5: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new FakeRTGILight(/*spawnPhysicalMarker = */ false));
                                mover->light->setIntensity(worldLight->getIntensity() * 10);
                                const auto worldLightColor = glm::vec3(0.0f, 1.0f, 0.0f);
                                mover->light->setColor(worldLightColor.r, worldLightColor.g, worldLightColor.b);
                                ((stratus::VirtualPointLight *)mover->light.get())->SetNumShadowSamples(9);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_6: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new FakeRTGILight(/*spawnPhysicalMarker = */ false));
                                mover->light->setIntensity(worldLight->getIntensity() * 10);
                                const auto worldLightColor = glm::vec3(0.0f, 0.0f, 1.0f);
                                mover->light->setColor(worldLightColor.r, worldLightColor.g, worldLightColor.b);
                                ((stratus::VirtualPointLight *)mover->light.get())->SetNumShadowSamples(9);
                                mover->position = camera->getPosition();
                                mover->addToScene();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_G: {
                            if (released) {
                                const bool enabled = World()->GetGlobalIlluminationEnabled();
                                World()->SetGlobalIlluminationEnabled( !enabled );
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
                        case SDL_SCANCODE_Z:
                            if (released) {
                                if (lightMovers.size() > 0) {
                                    lightMovers[lightMovers.size() - 1]->removeFromScene();
                                    lightMovers.pop_back();
                                }
                            }
                        default: break;
                    }
                    break;
                }
                default: break;
            }
        }

        stratus::RendererFrontend::Instance()->SetAtmosphericShadowing(fogDensity, scatterControl);

        // worldLight->setRotation(glm::vec3(75.0f, 0.0f, 0.0f));
        //worldLight->setRotation(stratus::Rotation(stratus::Degrees(30.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
        if (!worldLightPaused) {
            worldLight->offsetRotation(glm::vec3(worldLightMoveDirection * value * deltaSeconds, 0.0f, 0.0f));
        }

        #define LERP(x, v1, v2) (x * v1 + (1.0f - x) * v2)

        //renderer->toggleWorldLighting(worldLightEnabled);
        stratus::RendererFrontend::Instance()->SetWorldLight(worldLight);
        // worldLight->setColor(glm::vec3(1.0f, 0.75f, 0.5));
        // worldLight->setColor(glm::vec3(1.0f, 0.75f, 0.75f));
        const float x = std::sinf(stratus::Radians(worldLight->getRotation().x).value());
        worldLight->setColor(warmMorningColor);
        worldLight->setPosition(camera->getPosition());
        //worldLight->setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
        //renderer->setWorldLight(worldLight);

        // Check mouse state
        uint32_t buttonState = Window()->GetMouseStateLastFrame().mask;
        cameraSpeed.z = 0.0f;
        if ((buttonState & SDL_BUTTON_LMASK) != 0) { // left mouse button
            cameraSpeed.z = -camSpeed;
        }
        else if ((buttonState & SDL_BUTTON_RMASK) != 0) { // right mouse button
            cameraSpeed.z = camSpeed;
        }

        glm::vec3 tmpCamSpeed = cameraSpeed * camSpeedDivide;
        camera->setSpeed(tmpCamSpeed.y, tmpCamSpeed.z, tmpCamSpeed.x);
        camera->update(deltaSeconds);

        cameraLight->position = camera->getPosition();
        stratus::RendererFrontend::Instance()->SetClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        if (sponza) {
           sponza->SetLocalPosition(glm::vec3(0.0f));
           sponza->SetLocalScale(glm::vec3(15.0f));
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
    stratus::EntityPtr sponza;
    std::vector<stratus::EntityPtr> entities;
    stratus::CameraPtr camera;
    glm::vec3 cameraSpeed;
    float worldLightMoveDirection = 1.0; // -1.0 reverses it
    float camSpeedDivide = 0.25f; // For slowing camera down
    stratus::LightPtr cameraLight;
    stratus::InfiniteLightPtr worldLight;
    std::vector<std::unique_ptr<RandomLightMover>> lightMovers;
    bool camLightEnabled = true;
    bool worldLightPaused = true;
};

STRATUS_ENTRY_POINT(Sponza)