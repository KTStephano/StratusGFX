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

class Interrogation : public stratus::Application {
public:
    virtual ~Interrogation() = default;

    const char * GetAppName() const override {
        return "Interrogation";
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

        controller = stratus::InputHandlerPtr(new FrameRateController());
        Input()->AddInputHandler(controller);

        //const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        //controller = stratus::InputHandlerPtr(new WorldLightController(warmMorningColor));
        //Input()->AddInputHandler(controller);

        // Disable culling for this model since there are some weird parts that seem to be reversed
        stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../Resources/InterrogationRoom/scene.gltf", stratus::ColorSpace::SRGB, false, stratus::RenderFaceCulling::CULLING_NONE);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            interrogationRoom = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(interrogationRoom);
            //transform->SetLocalPosition(glm::vec3(0.0f));
            transform->SetLocalScale(glm::vec3(15.0f));
            INSTANCE(EntityManager)->AddEntity(interrogationRoom);
            received.push_back(e.GetPtr());
        });

        requested.push_back(e);

        INSTANCE(RendererFrontend)->SetFogColor(glm::vec3(167.0f / 255.0f, 166.0f / 255.0f, 157.0f / 255.0f));
        INSTANCE(RendererFrontend)->SetFogDensity(0.00125);

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
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f, 1.0f, 0.5f),
                                        1200.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f),
                                        1200.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_3: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f),
                                        1500.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_4: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f),
                                        2000.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_5: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f),
                                        3000.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_6: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f),
                                        6000.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_7: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f),
                                        12000.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_8: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f, 0.75f, 0.5f),
                                        24000.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_9: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f, 0.75f, 0.5f),
                                        48000.0f
                                    ),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_0: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f),
                                        65000.0f
                                    ),
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

        //worldLight->setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
        //renderer->setWorldLight(worldLight);

        stratus::RendererFrontend::Instance()->SetClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        if (requested.size() == received.size()) {
            requested.clear();

            LightCreator::CreateStationaryLight(
                LightParams(glm::vec3(-11.2298, 15.3294, 23.1447), glm::vec3(1, 1, 1), 1200, true),
                false
            );
            LightCreator::CreateStationaryLight(
                LightParams(glm::vec3(-17.5113, 15.3294, 19.8197), glm::vec3(1, 1, 1), 1200, true),
                false
            );
            LightCreator::CreateStationaryLight(
                LightParams(glm::vec3(-1.03776, 34.1635, -18.8183), glm::vec3(1, 1, 0.5), 1200, true),
                false
            );
        }

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
    stratus::EntityPtr interrogationRoom;
    std::vector<stratus::Async<stratus::Entity>> requested;
    std::vector<stratus::EntityPtr> received;
};

STRATUS_ENTRY_POINT(Interrogation)