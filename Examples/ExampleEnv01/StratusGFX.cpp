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

    RandomLightMover(const bool spawnPhysicalMarker = true) 
        : spawnPhysicalMarker(spawnPhysicalMarker) {
        cube = stratus::Application::Resources()->CreateCube();
        cube->GetRenderNode()->SetMaterial(stratus::Application::Materials()->CreateDefault());
        cube->GetRenderNode()->EnableLightInteraction(false);
        //cube->scale = glm::vec3(0.25f, 0.25f, 0.25f);
        cube->SetLocalScale(glm::vec3(1.0f));
        light = stratus::LightPtr(new stratus::PointLight());
        _changeDirection();
    }

    void addToScene() const {
        if (spawnPhysicalMarker) {
            stratus::Application::World()->AddStaticEntity(cube);
        }
        //r.addPointLight(light.get());
        stratus::Application::World()->AddLight(light);
    }

    void removeFromScene() const {
        stratus::Application::World()->RemoveEntity(cube);
        //r.addPointLight(light.get());
        stratus::Application::World()->RemoveLight(light);
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
    StationaryLight(const bool spawnPhysicalMarker = true) : RandomLightMover(spawnPhysicalMarker) {}

    void update(double deltaSeconds) override {
        cube->SetLocalPosition(position);
        light->position = position;
        stratus::MaterialPtr m = cube->GetRenderNode()->GetMeshContainer(0)->material;
        m->SetDiffuseColor(light->getColor());
    }
};

class StratusGFX : public stratus::Application {
public:
    virtual ~StratusGFX() = default;

    const char * GetAppName() const override {
        return "StratusGFX";
    }

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing StratusGFX" << std::endl;

        stratus::InputHandlerPtr controller(new CameraController());
        Input()->AddInputHandler(controller);

        const glm::vec3 warmMorningColor = glm::vec3(254.0f / 255.0f, 232.0f / 255.0f, 176.0f / 255.0f);
        controller = stratus::InputHandlerPtr(new WorldLightController(warmMorningColor));
        Input()->AddInputHandler(controller);

        //World()->SetAtmosphericShadowing(0.3f, 0.8f);

        // For textures see https://3dtextures.me/
        textures.push_back(Resources()->LoadTexture("../resources/textures/Substance_graph_BaseColor.jpg", true));
        textures.push_back(Resources()->LoadTexture("../resources/textures/Bark_06_basecolor.jpg", true));
        textures.push_back(Resources()->LoadTexture("../resources/textures/Wood_Wall_003_basecolor.jpg", true));
        textures.push_back(Resources()->LoadTexture("../resources/textures/Rock_Moss_001_basecolor.jpg", true));

        normalMaps.push_back(Resources()->LoadTexture("../resources/textures/Substance_graph_Normal.jpg", false));
        normalMaps.push_back(Resources()->LoadTexture("../resources/textures/Bark_06_normal.jpg", false));
        normalMaps.push_back(Resources()->LoadTexture("../resources/textures/Wood_Wall_003_normal.jpg", false));
        normalMaps.push_back(Resources()->LoadTexture("../resources/textures/Rock_Moss_001_normal.jpg", false));

        depthMaps.push_back(Resources()->LoadTexture("../resources/textures/Substance_graph_Height.png", false));
        depthMaps.push_back(Resources()->LoadTexture("../resources/textures/Bark_06_height.png", false));
        depthMaps.push_back(Resources()->LoadTexture("../resources/textures/Wood_Wall_003_height.png", false));
        depthMaps.push_back(Resources()->LoadTexture("../resources/textures/Rock_Moss_001_height.png", false));

        roughnessMaps.push_back(Resources()->LoadTexture("../resources/textures/Substance_graph_Roughness.jpg", false));
        roughnessMaps.push_back(Resources()->LoadTexture("../resources/textures/Bark_06_roughness.jpg", false));
        roughnessMaps.push_back(Resources()->LoadTexture("../resources/textures/Wood_Wall_003_roughness.jpg", false));
        roughnessMaps.push_back(Resources()->LoadTexture("../resources/textures/Rock_Moss_001_roughness.jpg", false));

        environmentMaps.push_back(Resources()->LoadTexture("../resources/textures/Substance_graph_AmbientOcclusion.jpg", true));
        environmentMaps.push_back(Resources()->LoadTexture("../resources/textures/Bark_06_ambientOcclusion.jpg", true));
        environmentMaps.push_back(Resources()->LoadTexture("../resources/textures/Wood_Wall_003_ambientOcclusion.jpg", true));
        environmentMaps.push_back(Resources()->LoadTexture("../resources/textures/Rock_Moss_001_ambientOcclusion.jpg", true));

        stratus::Async<stratus::Entity> e;
        e = Resources()->LoadModel("../resources/models/Latrine.fbx");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            outhouse = e.GetPtr(); 
            World()->AddStaticEntity(outhouse); 
            outhouse->SetLocalScale(glm::vec3(10.0f));
            outhouse->SetLocalPosition(glm::vec3(-50.0f, -10.0f, -45.0f));
        });

        e = Resources()->LoadModel("../resources/models/hromada_hlina_01_30k_f.FBX");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            clay = e.GetPtr(); 
            World()->AddStaticEntity(clay); 
            clay->SetLocalPosition(glm::vec3(100.0f, 0.0f, -50.0f));
        });

        e = Resources()->LoadModel("../resources/models/boubin_stump.FBX");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            stump = e.GetPtr(); 
            World()->AddStaticEntity(stump); 
            stump->SetLocalRotation(stratus::Rotation(stratus::Degrees(-180.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            stump->SetLocalPosition(glm::vec3(0.0f, -15.0f, -20.0f));
        });

        e = Resources()->LoadModel("../local/hintze-hall-1m.obj");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            hall = e.GetPtr(); 
            World()->AddStaticEntity(hall); 
            hall->SetLocalRotation(stratus::Rotation(stratus::Degrees(-90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            hall->SetLocalScale(glm::vec3(10.0f, 10.0f, 10.0f));
            hall->SetLocalPosition(glm::vec3(-250.0f, -30.0f, 0.0f));
        });

        e = Resources()->LoadModel("../local/model.obj");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            ramparts = e.GetPtr(); 
            World()->AddStaticEntity(ramparts); 
            ramparts->SetLocalPosition(glm::vec3(300.0f, 0.0f, -100.0f));
            ramparts->SetLocalRotation(stratus::Rotation(stratus::Degrees(90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            ramparts->SetLocalScale(glm::vec3(10.0f));
        });

        e = Resources()->LoadModel("../local/Rock_Terrain_SF.obj");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            rocks = e.GetPtr(); 
            World()->AddStaticEntity(rocks); 
            rocks->SetLocalPosition(glm::vec3(700.0f, -75.0f, -100.0f));
            rocks->SetLocalScale(glm::vec3(15.0f));
        });

        // Disable culling for this model since there are some weird parts that seem to be reversed
        e = Resources()->LoadModel("../local/sponza_scene/scene.gltf", stratus::RenderFaceCulling::CULLING_NONE);
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { 
            sponza = e.GetPtr(); 
            World()->AddStaticEntity(sponza); 
            sponza->SetLocalPosition(glm::vec3(0.0f, -300.0f, -500.0f));
            sponza->SetLocalScale(glm::vec3(15.0f));
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
            cube->GetRenderNode()->SetMaterial(mat);
            quad->GetRenderNode()->SetMaterial(mat);
            cubeMeshes.push_back(cube);
            quadMeshes.push_back(quad);
        }

        //quadMat.texture = Resources()->LoadTexture("../resources/textures/volcanic_rock_texture.png");
        srand(time(nullptr));
        for (int i = 0; i < 100; ++i) {
            size_t texIndex = rand() % textures.size();
            auto mesh = quadMeshes[texIndex]->Copy();
            mesh->SetLocalPosition(glm::vec3(rand() % 50, rand() % 50, rand() % 50));
            entities.push_back(mesh);
            mesh->SetLocalScale(glm::vec3(float(rand() % 5)));
            textureIndices.push_back(texIndex);
            World()->AddStaticEntity(mesh);
        }
        //std::vector<std::unique_ptr<Cube>> cubes;
        //cubeMat.texture = Resources()->LoadTexture("../resources/textures/wood_texture.jpg");
        for (int i = 0; i < 5000; ++i) {
            size_t texIndex = rand() % textures.size();
            auto mesh = cubeMeshes[texIndex]->Copy();
            entities.push_back(mesh);
            mesh->SetLocalPosition(glm::vec3(rand() % 3000, rand() % 50, rand() % 3000));
            mesh->SetLocalScale(glm::vec3(float(rand() % 25)));
            textureIndices.push_back(texIndex);
            World()->AddStaticEntity(mesh);
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
                        // case SDL_SCANCODE_UP: {
                        //     float brightness = worldLight->getIntensity() + lightIncreaseSpeed * deltaSeconds;
                        //     brightness = std::min(maxLightBrightness, brightness);
                        //     worldLight->setIntensity(brightness);
                        //     STRATUS_LOG << "Brightness: " << brightness << std::endl;
                        //     break;
                        // }
                        // case SDL_SCANCODE_DOWN: {
                        //     float brightness = worldLight->getIntensity() - lightIncreaseSpeed * deltaSeconds;
                        //     worldLight->setIntensity(brightness);
                        //     STRATUS_LOG << "Brightness: " << brightness << std::endl;
                        //     break;
                        // }
                        case SDL_SCANCODE_1: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight(/*spawnPhysicalMarker = */ false));
                                mover->light->setIntensity(worldLight->getIntensity() * 100);
                                const auto worldLightColor = worldLight->getColor();
                                mover->light->setColor(worldLightColor.r, worldLightColor.g, worldLightColor.b);
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

        //worldLight.setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
        //renderer->setWorldLight(worldLight);

        World()->SetClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

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
    std::vector<std::unique_ptr<RandomLightMover>> lightMovers;
};

STRATUS_ENTRY_POINT(StratusGFX)