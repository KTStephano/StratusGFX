#include "StratusCommon.h"
#include "StratusMath.h"
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
#include "StratusAsync.h"
#include <memory>
#include <filesystem>
#include "CameraController.h"
#include "WorldLightController.h"
#include "StratusEntityManager.h"
#include "StratusEntity.h"
#include "StratusEntityCommon.h"
#include "LightComponents.h"
#include "LightControllers.h"
#include "FrameRateController.h"

class StratusGFX : public stratus::Application {
public:
    virtual ~StratusGFX() = default;

    const char * GetAppName() const override {
        return "StratusGFX";
    }

    void PrintNodeHierarchy(const stratus::EntityPtr& p, const std::string& name, const std::string& prefix) {
        // auto rc = stratus::GetComponent<stratus::RenderComponent>(p);
        // std::cout << prefix << name << "{Meshes: " << (rc ? rc->GetMeshCount() : 0) << "}" << std::endl;
        // if (rc) {
        //     for (size_t i = 0; i < rc->GetMeshCount(); ++i) {
        //         std::cout << rc->GetMeshTransform(i) << std::endl;
        //     }
        // }
        // for (auto& c : p->GetChildNodes()) {
        //     PrintNodeHierarchy(c, name, prefix + "-> ");
        // }
    }

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing StratusGFX" << std::endl;

        LightCreator::Initialize();

        stratus::InputHandlerPtr controller(new CameraController());
        Input()->AddInputHandler(controller);

        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        auto wc = new WorldLightController(warmMorningColor, warmMorningColor, 5);
        wc->SetRotation(stratus::Rotation(stratus::Degrees(35.0f), stratus::Degrees(10.0f), stratus::Degrees(0)));
        controller = stratus::InputHandlerPtr(wc);
        Input()->AddInputHandler(controller);

        controller = stratus::InputHandlerPtr(new FrameRateController());
        Input()->AddInputHandler(controller);

        //World()->SetAtmosphericShadowing(0.3f, 0.8f);

        // For textures see https://3dtextures.me/
        textures.push_back(Resources()->LoadTexture("../Resources/resources/textures/Substance_graph_BaseColor.jpg", stratus::ColorSpace::SRGB));
        textures.push_back(Resources()->LoadTexture("../Resources/resources/textures/Bark_06_basecolor.jpg", stratus::ColorSpace::SRGB));
        textures.push_back(Resources()->LoadTexture("../Resources/resources/textures/Wood_Wall_003_basecolor.jpg", stratus::ColorSpace::SRGB));
        textures.push_back(Resources()->LoadTexture("../Resources/resources/textures/Rock_Moss_001_basecolor.jpg", stratus::ColorSpace::SRGB));

        normalMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Substance_graph_Normal.jpg", stratus::ColorSpace::LINEAR));
        normalMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Bark_06_normal.jpg", stratus::ColorSpace::LINEAR));
        normalMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Wood_Wall_003_normal.jpg", stratus::ColorSpace::LINEAR));
        normalMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Rock_Moss_001_normal.jpg", stratus::ColorSpace::LINEAR));

        depthMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Substance_graph_Height.png", stratus::ColorSpace::LINEAR));
        depthMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Bark_06_height.png", stratus::ColorSpace::LINEAR));
        depthMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Wood_Wall_003_height.png", stratus::ColorSpace::LINEAR));
        depthMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Rock_Moss_001_height.png", stratus::ColorSpace::LINEAR));

        roughnessMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Substance_graph_Roughness.jpg", stratus::ColorSpace::LINEAR));
        roughnessMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Bark_06_roughness.jpg", stratus::ColorSpace::LINEAR));
        roughnessMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Wood_Wall_003_roughness.jpg", stratus::ColorSpace::LINEAR));
        roughnessMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Rock_Moss_001_roughness.jpg", stratus::ColorSpace::LINEAR));

        environmentMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Substance_graph_AmbientOcclusion.jpg", stratus::ColorSpace::SRGB));
        environmentMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Bark_06_ambientOcclusion.jpg", stratus::ColorSpace::SRGB));
        environmentMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Wood_Wall_003_ambientOcclusion.jpg", stratus::ColorSpace::SRGB));
        environmentMaps.push_back(Resources()->LoadTexture("../Resources/resources/textures/Rock_Moss_001_ambientOcclusion.jpg", stratus::ColorSpace::SRGB));

        stratus::Async<stratus::Entity> e;
        e = Resources()->LoadModel("../Resources/resources/models/Latrine.fbx", stratus::ColorSpace::LINEAR, true, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            outhouse = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(outhouse);
            transform->SetLocalScale(glm::vec3(15.0f));
            transform->SetLocalPosition(glm::vec3(-50.0f, -10.0f, -45.0f));
            INSTANCE(EntityManager)->AddEntity(outhouse);
        });

        e = Resources()->LoadModel("../Resources/resources/models/hromada_hlina_01_30k_f.FBX", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            clay = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(clay);
            transform->SetLocalPosition(glm::vec3(100.0f, 0.0f, -50.0f));
            //transform->SetLocalRotation(stratus::Rotation(stratus::Degrees(-180.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            INSTANCE(EntityManager)->AddEntity(clay);
            PrintNodeHierarchy(clay, "Clay", "");
        });

        e = Resources()->LoadModel("../Resources/resources/models/boubin_stump.FBX", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            stump = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(stump);
            transform->SetLocalRotation(stratus::Rotation(stratus::Degrees(-180.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            transform->SetLocalPosition(glm::vec3(0.0f, -15.0f, -20.0f));
            INSTANCE(EntityManager)->AddEntity(stump);
            PrintNodeHierarchy(stump, "Stump", "");
        });

        e = Resources()->LoadModel("../Resources/local/hintze-hall-1m.obj", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            hall = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(hall);
            transform->SetLocalRotation(stratus::Rotation(stratus::Degrees(-90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            transform->SetLocalScale(glm::vec3(10.0f, 10.0f, 10.0f));
            transform->SetLocalPosition(glm::vec3(-250.0f, -30.0f, 0.0f));
            INSTANCE(EntityManager)->AddEntity(hall);
            PrintNodeHierarchy(hall, "Hall", "");
        });

        e = Resources()->LoadModel("../Resources/local/model.obj", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            ramparts = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(ramparts);
            transform->SetLocalPosition(glm::vec3(300.0f, 0.0f, -100.0f));
            transform->SetLocalRotation(stratus::Rotation(stratus::Degrees(90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            transform->SetLocalScale(glm::vec3(10.0f));
            INSTANCE(EntityManager)->AddEntity(ramparts);
        });

        e = Resources()->LoadModel("../Resources/local/Rock_Terrain_SF.obj", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            rocks = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(rocks);
            transform->SetLocalPosition(glm::vec3(700.0f, -75.0f, -100.0f));
            transform->SetLocalScale(glm::vec3(15.0f));
            INSTANCE(EntityManager)->AddEntity(rocks);
            PrintNodeHierarchy(rocks, "Rocks", "");
        });

        // Disable culling for this model since there are some weird parts that seem to be reversed
        e = Resources()->LoadModel("../Resources/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf", stratus::ColorSpace::SRGB, true, stratus::RenderFaceCulling::CULLING_CCW);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            sponza = e.GetPtr(); 
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(sponza);
            transform->SetLocalPosition(glm::vec3(0.0f, -300.0f, -500.0f));
            transform->SetLocalScale(glm::vec3(15.0f));
            INSTANCE(EntityManager)->AddEntity(sponza);
            PrintNodeHierarchy(sponza, "Sponza", "");
        });

        for (size_t texIndex = 0; texIndex < textures.size(); ++texIndex) {
            auto cube = Resources()->CreateCube();
            auto quad = Resources()->CreateQuad();
            stratus::MaterialPtr mat = Materials()->CreateMaterial("PrimitiveMat" + std::to_string(texIndex));
            mat->SetDiffuseTexture(textures[texIndex]);
            mat->SetNormalMap(normalMaps[texIndex]);
            mat->SetDepthMap(depthMaps[texIndex]);
            mat->SetRoughnessMap(roughnessMaps[texIndex]);
            mat->SetAmbientTexture(environmentMaps[texIndex]);
            stratus::RenderComponent * rc = stratus::GetComponent<stratus::RenderComponent>(cube);
            rc->SetMaterialAt(mat, 0);
            rc = stratus::GetComponent<stratus::RenderComponent>(quad);
            rc->SetMaterialAt(mat, 0);
            cubeMeshes.push_back(cube);
            quadMeshes.push_back(quad);
        }

        //quadMat.texture = Resources()->LoadTexture("../Resources/resources/textures/volcanic_rock_texture.png");
        srand(time(nullptr));
        for (int i = 0; i < 100; ++i) {
            size_t texIndex = rand() % textures.size();
            auto mesh = quadMeshes[texIndex]->Copy();
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(mesh);
            transform->SetLocalPosition(glm::vec3(rand() % 50, rand() % 50, rand() % 50));
            entities.push_back(mesh);
            transform->SetLocalScale(glm::vec3(float(rand() % 5)));
            textureIndices.push_back(texIndex);
            INSTANCE(EntityManager)->AddEntity(mesh);
        }
        //std::vector<std::unique_ptr<Cube>> cubes;
        // cubeMat.texture = Resources()->LoadTexture("../Resources/resources/textures/wood_texture.jpg");
        for (int i = 0; i < 5000; ++i) {
            size_t texIndex = rand() % textures.size();
            auto mesh = cubeMeshes[texIndex]->Copy();
            auto transform = stratus::GetComponent<stratus::LocalTransformComponent>(mesh);
            entities.push_back(mesh);
            transform->SetLocalPosition(glm::vec3(rand() % 3000, rand() % 50, rand() % 3000));
            transform->SetLocalScale(glm::vec3(float(rand() % 25)));
            textureIndices.push_back(texIndex);
            INSTANCE(EntityManager)->AddEntity(mesh);
        }

        // Create the light movers
        // for (int x = 0; x < 3000; x += 150) {
        //     for (int y = 0; y < 3000; y += 150) {
        //         std::unique_ptr<RandomLightMover> mover(new StationaryLight());
        //         mover->light->setIntensity(500.0f);
        //         mover->position = glm::vec3(float(x),
        //                                     0.0f, // float(rand() % 200),
        //                                     float(y));
        //         lightMovers.push_back(std::move(mover));
        //     }
        // }

        persp = glm::perspective(glm::radians(90.0f), 640 / 480.0f, 0.25f, 1000.0f);

        bool running = true;

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(const double deltaSeconds) override {
        const float maxAmbientIntensity = 0.03;
        if (Engine()->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << " (" << (deltaSeconds * 1000.0) << " ms)" << std::endl;
        }

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        auto worldLight = World()->GetWorldLight();
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
                                World()->RecompileShaders();
                            }
                            break;
                        case SDL_SCANCODE_1: {
                            if (released) {
                                LightCreator::CreateRandomLightMover(
                                    LightParams(camera->getPosition(),
                                        worldLight->getColor(),
                                        1000.0f
                                    )
                                );
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                LightCreator::CreateStationaryLight(
                                    LightParams(camera->getPosition(),
                                        glm::vec3(1.0f),
                                        1000.0f
                                    )
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
                                    )
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
                                    )
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
                                    )
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
                                    )
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
                                    )
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
                                    )
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
                                    )
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
                                    )
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

        //worldLight.setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
        //renderer->setWorldLight(worldLight);

        World()->SetClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

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
    std::vector<stratus::TextureHandle> textures;
    std::vector<stratus::TextureHandle> normalMaps;
    std::vector<stratus::TextureHandle> depthMaps;
    std::vector<stratus::TextureHandle> roughnessMaps;
    std::vector<stratus::TextureHandle> environmentMaps;
    stratus::EntityPtr outhouse;
    stratus::EntityPtr clay;
    stratus::EntityPtr stump;
    stratus::EntityPtr hall;
    stratus::EntityPtr ramparts;
    stratus::EntityPtr rocks;
    stratus::EntityPtr sponza;
    std::vector<stratus::EntityPtr> cubeMeshes;
    std::vector<stratus::EntityPtr> quadMeshes;
    std::vector<stratus::EntityPtr> entities;
    std::vector<size_t> textureIndices;
    glm::mat4 persp;
};

STRATUS_ENTRY_POINT(StratusGFX)