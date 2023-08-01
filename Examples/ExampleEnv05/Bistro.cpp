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

static void setupDayTime() {
    int spawned = 0;
    for (int x = -150; x < 200; x += 50) {
        for (int y = 0; y < 150; y += 20) {
            for (int z = -400; z < -50; z += 50) {
                    ++spawned;
                    LightCreator::CreateVirtualPointLight(
                        LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 1.0f),
                        false
                    );
            } 
        }
    }

    for (int x = -200; x < 95; x += 30) {
        for (int y = 0; y < 150; y += 15) {
            for (int z = -50; z < 200; z += 30) {
                    ++spawned;
                    LightCreator::CreateVirtualPointLight(
                        LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 1.0f),
                        false
                    );
            }
        }
    }

    for (int x = 300; x < 555; x += 30) {
        for (int y = 0; y < 50; y += 10) {
            for (int z = 150; z < 400; z += 30) {
                ++spawned;
                LightCreator::CreateVirtualPointLight(
                    LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 1.0f),
                    false
                );
            }
        }
    }

    for (int x = 180; x < 310; x += 30) {
        for (int y = 0; y < 160; y += 10) {
            for (int z = 100; z < 265; z += 30) {
                ++spawned;
                LightCreator::CreateVirtualPointLight(
                    LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 1.0f),
                    false
                );
            }
        }
    }

    for (int x = 240; x < 340; x += 30) {
        for (int y = 0; y < 160; y += 10) {
            for (int z = 130; z < 180; z += 30) {
                ++spawned;
                LightCreator::CreateVirtualPointLight(
                    LightParams(glm::vec3(float(x), float(y), float(z)), glm::vec3(1.0f), 1.0f),
                    false
                );
            }
        }
    }

    for (int x = -270; x < -160; x += 30) {
        for (int y = 0; y < 160; y += 10) {
            ++spawned;
            LightCreator::CreateVirtualPointLight(
                LightParams(glm::vec3(float(x), float(y), -250.0f), glm::vec3(1.0f), 1.0f),
                false
            );
        }
    }

    auto settings = INSTANCE(RendererFrontend)->GetSettings();
    settings.SetFogDensity(0.0f);
    settings.SetFogColor(glm::vec3(0.5f));
    settings.SetSkyboxIntensity(3.0f);
    settings.SetEmissionStrength(0.0f);
    INSTANCE(RendererFrontend)->SetSettings(settings);
    INSTANCE(RendererFrontend)->GetWorldLight()->SetAtmosphericLightingConstants(0.0045f, 0.0065f);
    
    STRATUS_LOG << "SPAWNED " << spawned << " VPLS" << std::endl;
}

static void setupNightTime() {
    auto settings = INSTANCE(RendererFrontend)->GetSettings();
    INSTANCE(RendererFrontend)->GetWorldLight()->SetEnabled(false);
    settings.SetFogDensity(0.00075);
    settings.SetFogColor(glm::vec3(0.5, 0.5, 0.125));
    settings.SetSkyboxIntensity(0.025);
    settings.SetEmissionStrength(5.0f);
    settings.SetEmissiveTextureMultiplier(5.0f);
    INSTANCE(RendererFrontend)->SetSettings(settings);

    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-12.0833, 24.51, -48.1222), glm::vec3(1, 1, 0.5), 800, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(2.13202, 24.51, -73.0153), glm::vec3(1, 1, 0.5), 800, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-18.6798, 26.02, -22.1233), glm::vec3(1, 1, 0.5), 800, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-5.15052, 26.385, 7.78559), glm::vec3(1, 1, 0.5), 800, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(16.6873, 24.355, 22.819), glm::vec3(1, 1, 0.5), 800, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(44.5133, 24.355, 33.9222), glm::vec3(1, 1, 0.5), 800, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(72.1779, 24.355, 44.6474), glm::vec3(1, 1, 0.5), 800, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(78.7009, 44.73, 86.5537), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-17.3537, 44.835, 100.604), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-21.2039, 44.835, 47.2472), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-43.6241, 44.835, -41.414), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-133.528, 41.2, 12.5846), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-172.268, 41.2, -37.5382), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-119.783, 41.2, -95.6311), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-177.766, 41.2, -149.001), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-201.141, 41.2, -72.1625), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-245.44, 41.2, -117.601), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-284.728, 33.875, -171.218), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-269.03, 33.875, -210.122), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-208.976, 45.175, -182.813), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-84.087, 41.425, -113.386), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-21.3645, 43.29, -182.658), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(179.52, 45.815, -415.614), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(176.673, 45.815, -344.445), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(89.9812, 46.37, -201.47), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(37.5956, 44.79, -212.5), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(215.365, 44.915, 187.586), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(244.89, 47.01, 231.121), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(299.942, 41.535, 135.591), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(332.324, 44.89, 240.823), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(350.854, 44.89, 183.909), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(208.095, 41.165, 127.712), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(492.692, 44.965, 343.702), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(388.343, 44.965, 338.671), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(15.0234, 41.1151, -105.322), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(3.22353, 41.9701, -115.278), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-7.2734, 44.1301, -125.024), glm::vec3(0.380392, 0.180392, 0.219608), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-18.3061, 45.8751, -135.402), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-3.00097, 40.6001, -69.9564), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-14.2764, 40.0951, -78.0438), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-39.0566, 40.5851, -96.4366), glm::vec3(0.380392, 0.180392, 0.219608), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-51.2181, 42.2451, -105.452), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-18.4518, 39.5201, -52.1642), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-29.844, 38.7851, -58.8508), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-43.8406, 38.5501, -66.5997), glm::vec3(0.380392, 0.180392, 0.219608), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-57.1667, 39.2851, -74.1774), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-26.1104, 37.4851, -21.7304), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-41.6363, 35.7901, -16.6856), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-57.5178, 35.6151, -11.3655), glm::vec3(0.380392, 0.180392, 0.219608), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-73.254, 35.6151, -5.92973), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-87.4548, 36.1851, -1.02524), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-96.7772, 44.93, 20.1176), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-4.29871, 41.5651, 8.65284), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-16.6329, 38.7701, 17.3138), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-29.2897, 37.535, 26.1631), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-40.8008, 37.17, 34.2195), glm::vec3(0.380392, 0.180392, 0.219608), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-53.3953, 37.8401, 43.1874), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(8.39885, 41.3901, 17.6688), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-0.811884, 41.1201, 30.0256), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-10.2191, 41.5651, 42.7579), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-18.4492, 42.4701, 53.6448), glm::vec3(0.380392, 0.180392, 0.219608), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(38.9259, 39.0101, 37.8689), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(34.3236, 36.6151, 47.8035), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(29.438, 34.7701, 59.3435), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(24.131, 34.0001, 70.9984), glm::vec3(0.380392, 0.180392, 0.219608), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(18.8899, 34.0001, 82.8694), glm::vec3(0.0470588, 0.356863, 0.054902), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(7.4249, 35.7752, 106.202), glm::vec3(0.380392, 0.180392, 0.219608), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(12.6971, 34.1802, 94.9015), glm::vec3(0.941176, 0.156863, 0.941176), 300, false),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(8.70603, 41.1401, -82.8636), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(20.1393, 41.1401, -104.487), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-1.21708, 41.1401, -63.8615), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(-12.2171, 41.1401, -43.1852), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(12.6869, 41.1951, 19.8702), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(35.821, 41.1951, 28.8144), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(55.7428, 41.1951, 36.5482), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
    LightCreator::CreateStationaryLight(
        LightParams(glm::vec3(77.314, 41.1951, 45.3662), glm::vec3(1, 1, 0.5), 400, true),
        false
    );
}

class Bistro : public stratus::Application {
public:
    virtual ~Bistro() = default;

    const char * GetAppName() const override {
        return "Bistro";
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

        //INSTANCE(Engine)->SetMaxFrameRate(30);
        //INSTANCE(RendererFrontend)->SetVsyncEnabled(true);

        LightCreator::Initialize();

        stratus::InputHandlerPtr controller(new CameraController());
        INSTANCE(InputManager)->AddInputHandler(controller);
        INSTANCE(RendererFrontend)->GetCamera()->SetPosition(glm::vec3(-117.849f, 17.9663f, -0.672086f));
        INSTANCE(RendererFrontend)->GetCamera()->SetAngle(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(-93.0f), stratus::Degrees(0.0f)));

        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        const glm::vec3 defaultSunColor = glm::vec3(1.0f);
        WorldLightController * wc = new WorldLightController(warmMorningColor, warmMorningColor, 10);
        wc->SetRotation(stratus::Rotation(stratus::Degrees(29.9668f), stratus::Degrees(10.0f), stratus::Degrees(0)));
        controller = stratus::InputHandlerPtr(wc);
        INSTANCE(InputManager)->AddInputHandler(controller);

        controller = stratus::InputHandlerPtr(new FrameRateController());
        INSTANCE(InputManager)->AddInputHandler(controller);

        // Alpha testing doesn't work so well for this scene 
        INSTANCE(RendererFrontend)->GetWorldLight()->SetAlphaTest(false);
        //INSTANCE(RendererFrontend)->GetWorldLight()->SetDepthBias(-0.001f);
        INSTANCE(RendererFrontend)->GetWorldLight()->SetDepthBias(0.0f);

        //const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        //controller = stratus::InputHandlerPtr(new WorldLightController(warmMorningColor));
        //INSTANCE(InputManager)->AddInputHandler(controller);

        // Disable culling for this model since there are some weird parts that seem to be reversed
        //stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../Resources/Bistro_v5_2/BistroExterior.fbx", stratus::ColorSpace::SRGB, stratus::RenderFaceCulling::CULLING_CCW);
        stratus::Async<stratus::Entity> e = stratus::ResourceManager::Instance()->LoadModel("../Resources/BistroGltf/Bistro.gltf", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            if (e.Failed()) return;
            bistro = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(bistro);
            //transform->SetLocalPosition(glm::vec3(0.0f));
            transform->SetLocalScale(glm::vec3(10.0f));
            INSTANCE(EntityManager)->AddEntity(bistro);
        });

        auto settings = INSTANCE(RendererFrontend)->GetSettings();
        settings.skybox = stratus::ResourceManager::Instance()->LoadCubeMap("../Resources/Skyboxes/learnopengl/sbox_", stratus::ColorSpace::NONE, "jpg");
        INSTANCE(RendererFrontend)->SetSettings(settings);

        bool running = true;

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(const double deltaSeconds) override {
        if (INSTANCE(Engine)->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << " (" << (deltaSeconds * 1000.0) << " ms)" << std::endl;
        } 

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        auto camera = INSTANCE(RendererFrontend)->GetCamera();
        auto worldLight = INSTANCE(RendererFrontend)->GetWorldLight();
        const glm::vec3 worldLightColor = worldLight->GetColor();
        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);

        // Check for key/mouse events
        auto events = INSTANCE(InputManager)->GetInputEventsLastFrame();
        for (auto& e : events) {
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
                        case SDL_SCANCODE_N:
                            if (released) {
                                daytime = !daytime;
                                if (daytime) setupDayTime();
                                else setupNightTime();
                            }
                            break;
                        case SDL_SCANCODE_1: {
                            if (released) {
                                LightCreator::CreateVirtualPointLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), worldLightColor, 100.0f)
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                LightCreator::CreateVirtualPointLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), worldLightColor, 50.0f)
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_3: {
                            if (released) {
                                LightCreator::CreateVirtualPointLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), worldLightColor, 15.0f)
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_4: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(1.0f, 1.0f, 0.5f), 800.0f),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_5: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(1.0f, 1.0f, 0.5f), 400.0f),
                                    false
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_6: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(12.0f / 255.0f, 91.0f / 255.0f, 14.0f / 255.0f), 300.0f, false),
                                    true
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_7: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(240.0f / 255.0f, 40.0f / 255.0f, 240.0f / 255.0f), 300.0f, false),
                                    true
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_8: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(INSTANCE(RendererFrontend)->GetCamera()->GetPosition(), glm::vec3(97.0f / 255.0f, 46.0f / 255.0f, 56.0f / 255.0f), 300.0f, false),
                                    true
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

        if (bistro != nullptr) {
            bistro = nullptr;
            setupDayTime();
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
    stratus::EntityPtr bistro;
    bool daytime = true;
};

STRATUS_ENTRY_POINT(Bistro)