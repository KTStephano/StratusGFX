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

class Test : public stratus::Application {
public:
    virtual ~Test() = default;

    const char * GetAppName() const override {
        return "Test";
    }

    void PrintNodeHierarchy(const stratus::EntityPtr& p, const std::string& name, const std::string& prefix) {
        auto rc = stratus::GetComponent<stratus::RenderComponent>(p);
        std::cout << prefix << name << "{Meshes: " << (rc ? rc->GetMeshCount() : 0) << "}" << std::endl;
        if (rc) {
            for (size_t i = 0; i < rc->GetMeshCount(); ++i) {
                std::cout << rc->GetMeshTransform(i) << std::endl;
            }
        }
        for (auto& c : p->GetChildNodes()) {
            PrintNodeHierarchy(c, name, prefix + "-> ");
        }
    }

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing " << GetAppName() << std::endl;

        LightCreator::Initialize();

        stratus::InputHandlerPtr controller(new CameraController());
        Input()->AddInputHandler(controller);

        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        const glm::vec3 defaultSunColor = glm::vec3(1.0f);
        auto wc = new WorldLightController(defaultSunColor, warmMorningColor, 10);
        //wc->SetRotation(stratus::Rotation(stratus::Degrees(123.991f), stratus::Degrees(10.0f), stratus::Degrees(0)));
        controller = stratus::InputHandlerPtr(wc);
        Input()->AddInputHandler(controller);

        controller = stratus::InputHandlerPtr(new FrameRateController());
        Input()->AddInputHandler(controller);

        INSTANCE(RendererFrontend)->GetWorldLight()->SetAlphaTest(true);
        INSTANCE(RendererFrontend)->GetWorldLight()->SetNumAtmosphericSamplesPerPixel(64);  

        //const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        //controller = stratus::InputHandlerPtr(new WorldLightController(warmMorningColor));
        //Input()->AddInputHandler(controller);

        // Disable culling for this model since there are some weird parts that seem to be reversed
        stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../Resources/deccer/scene.gltf", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_NONE);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) {  
            Test = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(Test);
            //transform->SetLocalPosition(glm::vec3(0.0f));
            //transform->SetLocalScale(glm::vec3(5.0));
            //transform->SetLocalRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(90.0f), stratus::Degrees(0.0f)));
            INSTANCE(EntityManager)->AddEntity(Test);
        });

        auto settings = INSTANCE(RendererFrontend)->GetSettings();
        settings.skybox = stratus::ResourceManager::Instance()->LoadCubeMap("../Resources/Skyboxes/learnopengl/sbox_", stratus::ColorSpace::LINEAR, "jpg");
        settings.SetSkyboxIntensity(0.001);
        settings.SetEmissionStrength(5.0f);
        settings.usePerceptualRoughness = false;
        INSTANCE(RendererFrontend)->SetSettings(settings);

        // INSTANCE(RendererFrontend)->SetFogDensity(0.00075);
        // INSTANCE(RendererFrontend)->SetFogColor(glm::vec3(0.5, 0.5, 0.125));

        bool running = true;

        const std::vector<float> ys = {-30.0f, 10.0f};
        const float offset = 60.0f; 

        for (float y : ys) {
            for (float x = -10.0f; x < 60.0f; x += offset) {
                for (float z = -60.0f; z < 60.0f; z += offset) {
                    const glm::vec3 location(x, y, z);
                    LightCreator::CreateStationaryLight(
                        LightParams(location, glm::vec3(0.941176, 0.156863, 0.941176), 100, false),
                        false
                    );

                    LightCreator::CreateStationaryLight(
                        LightParams(location, glm::vec3(0.380392, 0.180392, 0.219608), 100, false),
                        false
                    );

                    LightCreator::CreateStationaryLight(
                        LightParams(location, glm::vec3(0.0470588, 0.356863, 0.054902), 100, false),
                        false
                    );

                    // LightCreator::CreateStationaryLight(
                    //     LightParams(location, glm::vec3(1.0), 100, false),
                    //     false
                    // );
                }
            }
        }

        for (float y = -40.0f; y < 60.0f; y += offset) {
            for (float z = -60.0f; z < 60.0f; z += offset) {
                const glm::vec3 location(-15.0f, y, z);
                LightCreator::CreateStationaryLight(
                    LightParams(location, glm::vec3(0.941176, 0.156863, 0.941176), 100, false),
                    false
                );

                LightCreator::CreateStationaryLight(
                    LightParams(location, glm::vec3(0.380392, 0.180392, 0.219608), 100, false),
                    false
                );

                LightCreator::CreateStationaryLight(
                    LightParams(location, glm::vec3(0.0470588, 0.356863, 0.054902), 100, false),
                    false
                );

                // LightCreator::CreateStationaryLight(
                //     LightParams(location, glm::vec3(1.0), 100, false),
                //     false
                // );
            }
        }

        for (float y = -40.0f; y < 60.0f; y += 30.0f) {
            for (float x = -60.0f; x < 60.0f; x += 30.0f) {
                const glm::vec3 location(x, y, 15.0f);
                LightCreator::CreateStationaryLight(
                    LightParams(location, glm::vec3(1.0), 300, false),
                    false
                );
            }
        }

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(const double deltaSeconds) override {
        if (Engine()->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << " (" << (deltaSeconds * 1000.0) << " ms)" << std::endl;
        }

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        auto camera = World()->GetCamera();

        // Check for key/mouse events
        auto events = Input()->GetInputEventsLastFrame();
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
                                stratus::RendererFrontend::Instance()->RecompileShaders();
                            }
                            break;
                        case SDL_SCANCODE_1: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(World()->GetCamera()->GetPosition(), glm::vec3(224.0f / 255.0f, 157.0f / 255.0f, 55.0f / 255.0f), 10.0f),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(World()->GetCamera()->GetPosition(), glm::vec3(1.0), 10.0f),
                                    false
                                );
                            }
                            break;
                        }
                        default: break;
                    }
                    break;
                }
                default: break;
            }
        }

        if (Test != nullptr) {
            Test = nullptr;
            int spawned = 0;

            //for (int x = -16; x < 16; x += 5) {
            //   for (int y = 1; y < 18; y += 5) {
            //       for (int z = -15; z < 15; z += 5) {
            //           ++spawned;
            //           LightCreator::CreateVirtualPointLight(
            //               LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 100.0f),
            //               false
            //           );
            //       }
            //   }
            //}

            STRATUS_LOG << "SPAWNED " << spawned << " VPLS" << std::endl;
        }

        //worldLight->setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
        //renderer->setWorldLight(worldLight);

        stratus::RendererFrontend::Instance()->SetClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

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
    stratus::EntityPtr Test;
};

STRATUS_ENTRY_POINT(Test)