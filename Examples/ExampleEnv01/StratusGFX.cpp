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
        stratus::RendererFrontend::Instance()->AddStaticEntity(cube);
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
    StationaryLight() : RandomLightMover() {}

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

    std::string GetAppName() const {
        return "StratusGFX";
    }

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing StratusGFX" << std::endl;

        camera = stratus::CameraPtr(new stratus::Camera());
        stratus::RendererFrontend::Instance()->SetCamera(camera);

        // For textures see https://3dtextures.me/
        textures.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Substance_graph_BaseColor.jpg", true));
        textures.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Bark_06_basecolor.jpg", true));
        textures.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Wood_Wall_003_basecolor.jpg", true));
        textures.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Rock_Moss_001_basecolor.jpg", true));

        normalMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Substance_graph_Normal.jpg", false));
        normalMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Bark_06_normal.jpg", false));
        normalMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Wood_Wall_003_normal.jpg", false));
        normalMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Rock_Moss_001_normal.jpg", false));

        depthMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Substance_graph_Height.png", false));
        depthMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Bark_06_height.png", false));
        depthMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Wood_Wall_003_height.png", false));
        depthMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Rock_Moss_001_height.png", false));

        roughnessMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Substance_graph_Roughness.jpg", false));
        roughnessMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Bark_06_roughness.jpg", false));
        roughnessMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Wood_Wall_003_roughness.jpg", false));
        roughnessMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Rock_Moss_001_roughness.jpg", false));

        environmentMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Substance_graph_AmbientOcclusion.jpg", true));
        environmentMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Bark_06_ambientOcclusion.jpg", true));
        environmentMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Wood_Wall_003_ambientOcclusion.jpg", true));
        environmentMaps.push_back(stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/Rock_Moss_001_ambientOcclusion.jpg", true));

        stratus::Async<stratus::Entity> e;
        e = stratus::ResourceManager::Instance()->LoadModel("../resources/models/Latrine.fbx");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { outhouse = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(outhouse); });

        e = stratus::ResourceManager::Instance()->LoadModel("../resources/models/hromada_hlina_01_30k_f.FBX");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { clay = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(clay); });

        e = stratus::ResourceManager::Instance()->LoadModel("../resources/models/boubin_stump.FBX");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { stump = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(stump); });

        e = stratus::ResourceManager::Instance()->LoadModel("../local/hintze-hall-1m.obj");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { hall = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(hall); });

        e = stratus::ResourceManager::Instance()->LoadModel("../local/model.obj");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { ramparts = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(ramparts); });

        e = stratus::ResourceManager::Instance()->LoadModel("../local/Rock_Terrain_SF.obj");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { rocks = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(rocks); });

        e = stratus::ResourceManager::Instance()->LoadModel("../local/sponza_scene/scene.gltf");
        e.AddCallback([this](stratus::Async<stratus::Entity> e) { sponza = e.GetPtr(); stratus::RendererFrontend::Instance()->AddStaticEntity(sponza); });

        for (size_t texIndex = 0; texIndex < textures.size(); ++texIndex) {
            auto cube = stratus::ResourceManager::Instance()->CreateCube();
            auto quad = stratus::ResourceManager::Instance()->CreateQuad();
            stratus::MaterialPtr mat = stratus::MaterialManager::Instance()->CreateMaterial("PrimitiveMat" + std::to_string(texIndex));
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

        //quadMat.texture = stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/volcanic_rock_texture.png");
        srand(time(nullptr));
        for (int i = 0; i < 100; ++i) {
            size_t texIndex = rand() % textures.size();
            auto mesh = quadMeshes[texIndex]->Copy();
            mesh->SetLocalPosition(glm::vec3(rand() % 50, rand() % 50, rand() % 50));
            entities.push_back(mesh);
            mesh->SetLocalScale(glm::vec3(float(rand() % 5)));
            textureIndices.push_back(texIndex);
            stratus::RendererFrontend::Instance()->AddStaticEntity(mesh);
        }
        //std::vector<std::unique_ptr<Cube>> cubes;
        //cubeMat.texture = stratus::ResourceManager::Instance()->LoadTexture("../resources/textures/wood_texture.jpg");
        for (int i = 0; i < 5000; ++i) {
            size_t texIndex = rand() % textures.size();
            auto mesh = cubeMeshes[texIndex]->Copy();
            entities.push_back(mesh);
            mesh->SetLocalPosition(glm::vec3(rand() % 3000, rand() % 50, rand() % 3000));
            mesh->SetLocalScale(glm::vec3(float(rand() % 25)));
            textureIndices.push_back(texIndex);
            stratus::RendererFrontend::Instance()->AddStaticEntity(mesh);
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

        cameraLight = stratus::LightPtr(new stratus::PointLight());
        cameraLight->setCastsShadows(false);
        cameraLight->setIntensity(1200.0f);

        if (camLightEnabled) {
            stratus::RendererFrontend::Instance()->AddLight(cameraLight);
        }

        worldLight.setRotation(stratus::Rotation(stratus::Degrees(0.0f), stratus::Degrees(10.0f), stratus::Degrees(0.0f)));

        return true;
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(double deltaSeconds) override {
        float value = 1.0f;
        if (stratus::Engine::Instance()->FrameCount() % 100 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << std::endl;
        }
        const float camSpeed = 100.0f;
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
                                worldLightEnabled = !worldLightEnabled;
                            }
                            break;
                        case SDL_SCANCODE_P:
                            if (released) {
                                worldLightPaused = !worldLightPaused;
                            }
                            break;
                        case SDL_SCANCODE_UP:
                            worldLightBrightness += lightIncreaseSpeed * deltaSeconds;
                            worldLightBrightness = std::min(maxLightBrightness, worldLightBrightness);
                            STRATUS_LOG << "Brightness: " << worldLightBrightness << std::endl;
                            break;
                        case SDL_SCANCODE_DOWN:
                            worldLightBrightness -= lightIncreaseSpeed * deltaSeconds;
                            worldLightBrightness = std::max(0.0f, worldLightBrightness);
                            STRATUS_LOG << "Brightness: " << worldLightBrightness << std::endl;
                            break;
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

        // worldLight.setRotation(glm::vec3(75.0f, 0.0f, 0.0f));
        //worldLight.setRotation(stratus::Rotation(stratus::Degrees(30.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
        if (!worldLightPaused) {
            worldLight.offsetRotation(glm::vec3(value * deltaSeconds, 0.0f, 0.0f));
        }

        //renderer->toggleWorldLighting(worldLightEnabled);
        stratus::RendererFrontend::Instance()->SetWorldLightingEnabled(worldLightEnabled);
        // worldLight.setColor(glm::vec3(1.0f, 0.75f, 0.5));
        // worldLight.setColor(glm::vec3(1.0f, 0.75f, 0.75f));
        worldLight.setColor(glm::vec3(1.0f));
        worldLight.setIntensity(worldLightBrightness);
        worldLight.setPosition(camera->getPosition());
        //worldLight.setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
        //renderer->setWorldLight(worldLight);
        stratus::RendererFrontend::Instance()->SetWorldLightColor(worldLight.getColor());
        stratus::RendererFrontend::Instance()->SetWorldLightIntensity(worldLightBrightness);
        stratus::RendererFrontend::Instance()->SetWorldLightRotation(worldLight.getRotation());

        // Check mouse state
        uint32_t buttonState = stratus::RendererFrontend::Instance()->GetMouseState().mask;
        cameraSpeed.z = 0.0f;
        if ((buttonState & SDL_BUTTON_LMASK) != 0) { // left mouse button
            cameraSpeed.z = -camSpeed;
        }
        else if ((buttonState & SDL_BUTTON_RMASK) != 0) { // right mouse button
            cameraSpeed.z = camSpeed;
        }

        camera->setSpeed(cameraSpeed.y, cameraSpeed.z, cameraSpeed.x);
        camera->update(deltaSeconds);

        cameraLight->position = camera->getPosition();
        stratus::RendererFrontend::Instance()->SetClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        if (outhouse) {
            outhouse->SetLocalScale(glm::vec3(10.0f));
            outhouse->SetLocalPosition(glm::vec3(-50.0f, -10.0f, -45.0f));
        }
        //renderer->addDrawable(outhouse);

        //clay.scale = glm::vec3(1.0f);
        //clay.rotation = glm::vec3(-90.0f, 0.0f, 0.0f);
        if (clay) {
            clay->SetLocalPosition(glm::vec3(100.0f, 0.0f, -50.0f));
        }
        //clay.rotation = stratus::Rotation(stratus::Degrees(-90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f));
        //renderer->addDrawable(clay);

        if (stump) {
            stump->SetLocalRotation(stratus::Rotation(stratus::Degrees(-180.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            stump->SetLocalPosition(glm::vec3(0.0f, -15.0f, -20.0f));
        }
        //renderer->addDrawable(stump);

        if (hall) {
            hall->SetLocalRotation(stratus::Rotation(stratus::Degrees(-90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            hall->SetLocalScale(glm::vec3(10.0f, 10.0f, 10.0f));
            hall->SetLocalPosition(glm::vec3(-250.0f, -30.0f, 0.0f));
        }
        //renderer->addDrawable(hall);

        if (ramparts) {
            ramparts->SetLocalPosition(glm::vec3(300.0f, 0.0f, -100.0f));
            ramparts->SetLocalRotation(stratus::Rotation(stratus::Degrees(90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
            ramparts->SetLocalScale(glm::vec3(10.0f));
        }
        //renderer->addDrawable(ramparts);

        if (rocks) {
            rocks->SetLocalPosition(glm::vec3(700.0f, -75.0f, -100.0f));
            rocks->SetLocalScale(glm::vec3(15.0f));
        }

        if (sponza) {
           sponza->SetLocalPosition(glm::vec3(0.0f, -300.0f, -500.0f));
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
    virtual void ShutDown() override {
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
    stratus::CameraPtr camera;
    glm::vec3 cameraSpeed;
    stratus::LightPtr cameraLight;
    stratus::InfiniteLight worldLight;
    std::vector<std::unique_ptr<RandomLightMover>> lightMovers;
    bool camLightEnabled = true;
    bool worldLightEnabled = true;
    bool worldLightPaused = true;
    float worldLightBrightness = 5.0f;
};

STRATUS_ENTRY_POINT(StratusGFX)