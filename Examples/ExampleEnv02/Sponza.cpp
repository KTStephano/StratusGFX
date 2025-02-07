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
#include "CameraController.h"
#include "WorldLightController.h"
#include "LightComponents.h"
#include "LightControllers.h"
#include "StratusTransformComponent.h"
#include "StratusGpuCommon.h"
#include "WorldLightController.h"
#include "FrameRateController.h"
#include "StratusWindow.h"

class Sponza : public stratus::Application {
public:
    virtual ~Sponza() = default;

    const char * GetAppName() const override {
        return "Sponza";
    }

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing " << GetAppName() << std::endl;

        //INSTANCE(Window)->SetWindowDims(1920, 1080);

        LightCreator::Initialize();

        stratus::InputHandlerPtr controller(new CameraController());
        cc = (CameraController*)controller.get();
        INSTANCE(InputManager)->AddInputHandler(controller);

        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        const glm::vec3 defaultSunColor = glm::vec3(1.0f);
        const glm::vec3 moonlightColor = glm::vec3(79.0f / 255.0f, 105.0f / 255.0f, 136.0f / 255.0f);
        auto wc = new WorldLightController(defaultSunColor, warmMorningColor, 15.0f);
        wlc = wc;
        //auto wc = new WorldLightController(moonlightColor, moonlightColor, 0.5f);
        wc->SetRotation(stratus::Rotation(stratus::Degrees(56.8385f), stratus::Degrees(0.0f), stratus::Degrees(0)));
        controller = stratus::InputHandlerPtr(wc);
        INSTANCE(InputManager)->AddInputHandler(controller);

        controller = stratus::InputHandlerPtr(new FrameRateController());
        INSTANCE(InputManager)->AddInputHandler(controller);

        {
            auto cube = INSTANCE(ResourceManager)->CreateCube();
            auto local = cube->Components().GetComponent<stratus::LocalTransformComponent>().component;
            auto rc = cube->Components().GetComponent<stratus::RenderComponent>().component;
            local->SetLocalPosition(glm::vec3(2.5335, 237.164, -25.085));
            local->SetLocalScale(glm::vec3(50.0f, 1.0f, 100.0f));
            rc->SetMaterialAt(INSTANCE(MaterialManager)->GetOrCreateMaterial("White"), 0);
            rc->GetMaterialAt(0)->SetDiffuseColor(glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
            //INSTANCE(EntityManager)->AddEntity(cube);

            cube = INSTANCE(ResourceManager)->CreateCube();
            local = cube->Components().GetComponent<stratus::LocalTransformComponent>().component;
            rc = cube->Components().GetComponent<stratus::RenderComponent>().component;
            local->SetLocalPosition(glm::vec3(2.5, 2.5, 100));
            local->SetLocalScale(glm::vec3(30.0f, 1.0f, 20.0f));
            rc->SetMaterialAt(INSTANCE(MaterialManager)->GetOrCreateMaterial("Red"), 0);
            rc->GetMaterialAt(0)->SetDiffuseColor(glm::vec4(0.75f, 0.1f, 0.1f, 1.0f));
            //INSTANCE(EntityManager)->AddEntity(cube);

            cube = INSTANCE(ResourceManager)->CreateCube();
            local = cube->Components().GetComponent<stratus::LocalTransformComponent>().component;
            rc = cube->Components().GetComponent<stratus::RenderComponent>().component;
            local->SetLocalPosition(glm::vec3(2.5, 2.5, 60));
            local->SetLocalScale(glm::vec3(30.0f, 1.0f, 20.0f));
            rc->SetMaterialAt(INSTANCE(MaterialManager)->GetOrCreateMaterial("Green"), 0);
            rc->GetMaterialAt(0)->SetDiffuseColor(glm::vec4(0.1f, 0.75f, 0.1f, 1.0f));
            //INSTANCE(EntityManager)->AddEntity(cube);

            cube = INSTANCE(ResourceManager)->CreateCube();
            local = cube->Components().GetComponent<stratus::LocalTransformComponent>().component;
            rc = cube->Components().GetComponent<stratus::RenderComponent>().component;
            local->SetLocalPosition(glm::vec3(2.5, 2.5, 20));
            local->SetLocalScale(glm::vec3(30.0f, 1.0f, 20.0f));
            rc->SetMaterialAt(INSTANCE(MaterialManager)->GetOrCreateMaterial("Blue"), 0);
            rc->GetMaterialAt(0)->SetDiffuseColor(glm::vec4(0.1f, 0.1f, 0.75f, 1.0f));
            //INSTANCE(EntityManager)->AddEntity(cube);

            cube = INSTANCE(ResourceManager)->CreateCube();
            local = cube->Components().GetComponent<stratus::LocalTransformComponent>().component;
            rc = cube->Components().GetComponent<stratus::RenderComponent>().component;
            local->SetLocalPosition(glm::vec3(2.5, 2.5, -20));
            local->SetLocalScale(glm::vec3(30.0f, 1.0f, 20.0f));
            rc->SetMaterialAt(INSTANCE(MaterialManager)->GetOrCreateMaterial("Purple"), 0);
            rc->GetMaterialAt(0)->SetDiffuseColor(glm::vec4(0.75f, 0.1f, 0.75f, 1.0f));
            //INSTANCE(EntityManager)->AddEntity(cube);

            cube = INSTANCE(ResourceManager)->CreateCube();
            local = cube->Components().GetComponent<stratus::LocalTransformComponent>().component;
            rc = cube->Components().GetComponent<stratus::RenderComponent>().component;
            local->SetLocalPosition(glm::vec3(2.5, 2.5, -60));
            local->SetLocalScale(glm::vec3(30.0f, 1.0f, 20.0f));
            rc->SetMaterialAt(INSTANCE(MaterialManager)->GetOrCreateMaterial("Yellow"), 0);
            rc->GetMaterialAt(0)->SetDiffuseColor(glm::vec4(0.75f, 0.75f, 0.1f, 1.0f));
            //INSTANCE(EntityManager)->AddEntity(cube);

            cube = INSTANCE(ResourceManager)->CreateCube();
            local = cube->Components().GetComponent<stratus::LocalTransformComponent>().component;
            rc = cube->Components().GetComponent<stratus::RenderComponent>().component;
            local->SetLocalPosition(glm::vec3(2.5, 2.5, -100));
            local->SetLocalScale(glm::vec3(30.0f, 1.0f, 20.0f));
            rc->SetMaterialAt(INSTANCE(MaterialManager)->GetOrCreateMaterial("Orange"), 0);
            rc->GetMaterialAt(0)->SetDiffuseColor(glm::vec4(0.75f, 0.5f, 0.1f, 1.0f));
            //INSTANCE(EntityManager)->AddEntity(cube);
        }

        // Moonlight
        //worldLight->setColor(glm::vec3(80.0f / 255.0f, 104.0f / 255.0f, 134.0f / 255.0f));
        //worldLight->setIntensity(0.5f);

        //INSTANCE(RendererFrontend)->SetAtmosphericShadowing(0.2f, 0.3f);

        // Disable culling for this model since there are some weird parts that seem to be reversed
        //stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../Resources/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../Resources/Sponza.glb", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        stratus::Async<stratus::Entity> e2 = stratus::ResourceManager::Instance()->LoadModel("../Resources/SponzaCurtains.glb", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        requested.push_back(e);
        requested.push_back(e2);
        
        auto callback = [this](stratus::Async<stratus::Entity> e) { 
            if (e.Failed()) return;
            //STRATUS_LOG << "Adding\n";
            received.push_back(e.GetPtr());
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(e.GetPtr());
            //transform->SetLocalPosition(glm::vec3(0.0f));
            //transform->SetLocalScale(glm::vec3(15.0f));
            transform->SetLocalScale(glm::vec3(15.0f));
            transform->SetLocalRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(90.0f), stratus::Degrees(0.0f)));  
            INSTANCE(EntityManager)->AddEntity(e.GetPtr());
            //INSTANCE(RendererFrontend)->AddDynamicEntity(sponza);
        };

        e.AddCallback(callback);
        e2.AddCallback(callback);

        auto settings = INSTANCE(RendererFrontend)->GetSettings();
        settings.skybox = stratus::ResourceManager::Instance()->LoadCubeMap("../Resources/Skyboxes/learnopengl/sbox_", stratus::ColorSpace::SRGB, "jpg");
        //settings.SetSkyboxIntensity(0.05f);
        //settings.SetSkyboxColorMask(moonlightColor);
        settings.SetAlphaDepthTestThreshold(0.75f);
        settings.taaEnabled = true;
        INSTANCE(RendererFrontend)->SetSettings(settings);

        INSTANCE(RendererFrontend)->GetWorldLight()->SetAlphaTest(true);

        bool running = true;

        // for (int i = 0; i < 64; ++i) {
        //     float x = rand() % 600;
        //     float y = rand() % 600;
        //     float z = rand() % 200;
        //     stratus::VirtualPointLight * vpl = new stratus::VirtualPointLight();
        //     vpl->setIntensity(worldLight->getIntensity() * 50.0f);
        //     vpl->position = glm::vec3(x, y, z);
        //     vpl->setColor(worldLight->getColor());
        //     INSTANCE(RendererFrontend)->AddLight(stratus::LightPtr((stratus::Light *)vpl));
        // }

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(const double deltaSeconds) override {
        if (INSTANCE(Engine)->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << " (" << (deltaSeconds * 1000.0) << " ms)" << std::endl;
        }

        auto worldLight = INSTANCE(RendererFrontend)->GetWorldLight();
        const glm::vec3 worldLightColor = worldLight->GetColor();
        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        // Check for key/mouse events
        auto events = INSTANCE(InputManager)->GetInputEventsLastFrame();
        for (auto e : events) {
            switch (e.type) {
                case SDL_QUIT:
                    return stratus::SystemStatus::SYSTEM_SHUTDOWN;
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
                        case SDL_SCANCODE_R:
                            if (released) {
                                INSTANCE(RendererFrontend)->RecompileShaders();
                            }
                            break;
                        case SDL_SCANCODE_1: {
                            if (released) {
                                //LightCreator::CreateStationaryLight(
                                //    //LightParams(INSTANCE(RendererFrontend)->GetCamera()->getPosition(), glm::vec3(1.0f, 1.0f, 0.5f), 1200.0f)
                                //    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), warmMorningColor, 600.0f),
                                //    false
                                //);
                                LightCreator::CreateVirtualPointLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(1.0f), 1.0f),
                                    true
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(1.0f, 1.0f, 0.5f), 800.0f),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_LSHIFT:
                            if (released) {
                                sunMovementUnlocked_ = !sunMovementUnlocked_;
                            }
                            break;
                        default: break;
                    }
                    break;
                }
                default: break;
            }
        }

        if (requested.size() == received.size()) {
           received.clear();
           int spawned = 0;
           for (int x = 80; x >= -80; x -= 20) {
              for (int y = 5; y <= 210; y += 20) {
                  for (int z = -160; z <= 230; z += 20) {
                          ++spawned;
                          LightCreator::CreateVirtualPointLight(
                              LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 1.0f),
                              false
                          );
                  }
              }
           }

           //for (int x = 10; x < 40; x += 5) {
           //    for (int y = 0; y < 10; y += 5) {
           //        for (int z = 125; z < 160; z += 10) {
           //            ++spawned;
           //            LightCreator::CreateVirtualPointLight(
           //                LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 1.0f),
           //                true
           //            );
           //        }
           //    }
           //}

        //    for (int x = -160; x < 150; x += 20) {
        //        for (int y = 15; y < 150; y += 20) {
        //            for (int z = -60; z < 60; z += 10) {
        //                    ++spawned;
        //                    LightCreator::CreateVirtualPointLight(
        //                        LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 100.0f),
        //                        true
        //                    );
        //            }
        //        }
        //    }

           STRATUS_LOG << "SPAWNED " << spawned << " VPLS\n";
        }

        if (sunMovementUnlocked_) {
            auto shift = cc->GetRecentMouseXYChange() * 5.0f * float(deltaSeconds);
            wlc->OffsetWorldLightRotation(glm::vec3(shift.x, 0.0f, 0.0));
        }

        // worldLight->setRotation(glm::vec3(75.0f, 0.0f, 0.0f));
        //worldLight->setRotation(stratus::Rotation(stratus::Degrees(30.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));

        #define LERP(x, v1, v2) (x * v1 + (1.0f - x) * v2)

        //renderer->toggleWorldLighting(worldLightEnabled);
        // worldLight->setColor(glm::vec3(1.0f, 0.75f, 0.5));
        // worldLight->setColor(glm::vec3(1.0f, 0.75f, 0.75f));
        //const float x = std::sinf(stratus::Radians(worldLight->getRotation().x).value());
        
        //worldLight->setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
        //renderer->setWorldLight(worldLight);

        INSTANCE(RendererFrontend)->SetClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        //renderer->addDrawable(rocks);

        // Add the camera's light
        //if (camLightEnabled) renderer->addPointLight(&cameraLight);
        //for (auto & entity : entities) {
        //    renderer->addDrawable(entity);
        //}

        //renderer->end(camera);

        //// 0 lets it run as fast as it can
        //SDL_GL_SetSwapInterval(0);
        //// Swap front and back buffer
        //SDL_GL_SwapWindow(window);

        return stratus::SystemStatus::SYSTEM_CONTINUE;
    }

    // Perform any resource cleanup
    virtual void Shutdown() override {
        LightCreator::Shutdown();
    }

private:
    std::vector<stratus::Async<stratus::Entity>> requested;
    std::vector<stratus::EntityPtr> received;
    bool sunMovementUnlocked_ = false;
    // Unsafe
    CameraController* cc;
    WorldLightController* wlc;
};

STRATUS_ENTRY_POINT(Sponza)