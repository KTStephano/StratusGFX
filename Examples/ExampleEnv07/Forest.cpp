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

class Forest : public stratus::Application {
public:
    virtual ~Forest() = default;

    const char * GetAppName() const override {
        return "Forest";
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
        controller = stratus::InputHandlerPtr(new WorldLightController(defaultSunColor, warmMorningColor, 5));
        Input()->AddInputHandler(controller);
        
        INSTANCE(RendererFrontend)->GetWorldLight()->SetAlphaTest(true);
        INSTANCE(RendererFrontend)->GetWorldLight()->SetNumAtmosphericSamplesPerPixel(64);

        //const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        //controller = stratus::InputHandlerPtr(new WorldLightController(warmMorningColor));
        //Input()->AddInputHandler(controller);

        // Disable culling for this model since there are some weird parts that seem to be reversed
        stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../../PineForest/scene.gltf", stratus::ColorSpace::SRGB, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            forest = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(forest);
            //transform->SetLocalPosition(glm::vec3(0.0f));
            transform->SetLocalScale(glm::vec3(10.0f));
            //transform->SetLocalRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(70.0f), stratus::Degrees(0.0f)));
            INSTANCE(EntityManager)->AddEntity(forest);
        });

        INSTANCE(RendererFrontend)->SetSkybox(stratus::ResourceManager::Instance()->LoadCubeMap("../resources/textures/Skyboxes/learnopengl/sbox_", stratus::ColorSpace::LINEAR, "jpg"));
        INSTANCE(RendererFrontend)->SetSkyboxIntensity(3.0f);

        bool running = true;

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
                                LightCreator::CreateVirtualPointLight(
                                    LightParams(World()->GetCamera()->getPosition(), glm::vec3(1.0f), 100.0f)
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                LightCreator::CreateVirtualPointLight(
                                    LightParams(World()->GetCamera()->getPosition(), glm::vec3(1.0f), 50.0f)
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

        if (forest != nullptr) {
            forest = nullptr;
            int spawned = 0;

            for (int x = -90; x < 90; x += 20) {
                //for (int y = 3; y < 10; y += 3) {
                    for (int z = -350; z < 50; z += 50) {
                        ++spawned;
                        LightCreator::CreateVirtualPointLight(
                            LightParams(glm::vec3(float(x), 20.0f, float(z)), glm::vec3(1.0f), 100.0f),
                            true
                        );
                    }
                //}
            }

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
    stratus::EntityPtr forest;
};

STRATUS_ENTRY_POINT(Forest)