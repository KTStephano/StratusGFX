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

class CornellBox : public stratus::Application {
public:
    virtual ~CornellBox() = default;

    const char * GetAppName() const override {
        return "CornellBox";
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
        INSTANCE(InputManager)->AddInputHandler(controller);

        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        const glm::vec3 defaultSunColor = glm::vec3(1.0f);
        const glm::vec3 sunsetColor = glm::vec3(251.0f / 255.0f, 144.0f / 255.0f, 98.0f / 255.0f);
        //const glm::vec3 sunsetColor = warmMorningColor;
        auto wc = new WorldLightController(defaultSunColor, defaultSunColor, 9.5f);
        //wc->SetRotation(stratus::Rotation(stratus::Degrees(123.991f), stratus::Degrees(10.0f), stratus::Degrees(0)));
        //wc->SetRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(29.6286f), stratus::Degrees(0.0f)));
        controller = stratus::InputHandlerPtr(wc);
        INSTANCE(InputManager)->AddInputHandler(controller);

        controller = stratus::InputHandlerPtr(new FrameRateController());
        INSTANCE(InputManager)->AddInputHandler(controller);

        INSTANCE(RendererFrontend)->GetWorldLight()->SetAlphaTest(true);
        INSTANCE(RendererFrontend)->GetWorldLight()->SetNumAtmosphericSamplesPerPixel(64);  
        INSTANCE(RendererFrontend)->GetWorldLight()->SetDepthBias(0.0f);

        //const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        //controller = stratus::InputHandlerPtr(new WorldLightController(warmMorningColor));
        //INSTANCE(InputManager)->AddInputHandler(controller);

        // Disable culling for this model since there are some weird parts that seem to be reversed
        stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../Resources/CornellBox.glb", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_NONE);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) {  
            if (e.Failed()) return;
            CornellBox = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(CornellBox);
            //transform->SetLocalPosition(glm::vec3(0.0f));
            transform->SetLocalScale(glm::vec3(15.0f));
            transform->SetLocalRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(-180.0f), stratus::Degrees(0.0f)));
            INSTANCE(EntityManager)->AddEntity(CornellBox);
        });

        auto settings = INSTANCE(RendererFrontend)->GetSettings();
        settings.skybox = stratus::ResourceManager::Instance()->LoadCubeMap("../Resources/Skyboxes/learnopengl/sbox_", stratus::ColorSpace::NONE, "jpg");
        settings.SetSkyboxIntensity(stratus::Float8Bit(255));
        settings.SetSkyboxColorMask(sunsetColor);
        settings.SetEmissionStrength(5.0f);
        settings.usePerceptualRoughness = false;
        settings.cascadeResolution = stratus::RendererCascadeResolution::CASCADE_RESOLUTION_8192;
        // Brighten the GI in the scene
        settings.SetMinGiOcclusionFactor(0.65f);
        settings.taaEnabled = false;
        INSTANCE(RendererFrontend)->SetSettings(settings);

        // INSTANCE(RendererFrontend)->SetFogDensity(0.00075);
        // INSTANCE(RendererFrontend)->SetFogColor(glm::vec3(0.5, 0.5, 0.125));

        bool running = true;

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(const double deltaSeconds) override {
        if (INSTANCE(Engine)->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << " (" << (deltaSeconds * 1000.0) << " ms)" << std::endl;
        }

        const float vplIntensity = 0.01; //0.05f / 1.0f;
        const bool showVpls = false;

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        auto camera = INSTANCE(RendererFrontend)->GetCamera();

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
                                stratus::RendererFrontend::Instance()->RecompileShaders();
                            }
                            break;
                            case SDL_SCANCODE_1: {
                                if (released) {
                                    LightCreator::CreateVirtualPointLight(
                                        LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(1.0f), vplIntensity),
                                        showVpls
                                    );
                                }
                                break;
                            }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(1.0), 10.0f),
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

        if (CornellBox != nullptr) {
            CornellBox = nullptr;
            int spawned = 0;

            for (float x = -14.0f; x <= 14.0f; x += 4.0f) {
                for (float y = 1; y < 30; y += 5.0f) {
                    for (float z = -16.0f; z <= 16.0f; z += 3.0f) {
                        ++spawned;
                        const glm::vec3 location(x, y, z);
                        LightCreator::CreateVirtualPointLight(
                            LightParams(location, glm::vec3(1.0f), vplIntensity),
                            showVpls
                        );
                    }
                }
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
    stratus::EntityPtr CornellBox;
};

STRATUS_ENTRY_POINT(CornellBox)