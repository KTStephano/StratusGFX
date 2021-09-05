#include "StratusCommon.h"
#include "glm/glm.hpp"
#include <iostream>
#include <StratusPipeline.h>
#include <StratusRenderer.h>
#include <StratusQuad.h>
#include <StratusCamera.h>
#include <chrono>
#include "StratusEngine.h"
#include "StratusLog.h"
#include <StratusCube.h>
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
    std::unique_ptr<stratus::RenderEntity> cube;
    std::unique_ptr<stratus::Light> light;
    glm::vec3 position;
    glm::vec3 speed;

    RandomLightMover() {
        cube = std::make_unique<stratus::RenderEntity>();
        cube->setLightProperties(stratus::LightProperties::FLAT);
        //cube->scale = glm::vec3(0.25f, 0.25f, 0.25f);
        cube->scale = glm::vec3(1.0f);
        cube->meshes.push_back(std::make_shared<stratus::Cube>());
        light = std::make_unique<stratus::PointLight>();
        speed = glm::vec3(float(rand() % 15 + 5));
        _changeDirection();
    }

    void addToScene(stratus::Renderer & r) const {
        r.addDrawable(cube.get());
        r.addPointLight(light.get());
    }

    virtual void update(double deltaSeconds) {
        position = position + speed * _direction * float(deltaSeconds);
        cube->position = position;
        light->position = position;
        stratus::RenderMaterial m = cube->meshes[0]->getMaterial();
        m.diffuseColor = light->getColor();
        cube->meshes[0]->setMaterial(m);

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
        cube->position = position;
        light->position = position;
        stratus::RenderMaterial m = cube->meshes[0]->getMaterial();
        m.diffuseColor = light->getColor();
        cube->meshes[0]->setMaterial(m);
    }
};

class StratusGFX : public stratus::Application {
public:
    virtual ~StratusGFX() = default;

    // Perform first-time initialization - true if success, false otherwise
    virtual bool Initialize() override {
        STRATUS_LOG << "Initializing StratusGFX" << std::endl;

        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            STRATUS_ERROR << "Unable to initialize sdl2" << std::endl;
            return -1;
        }

        window = SDL_CreateWindow("StratusGFX",
                100, 100, // location x/y on screen
                1920, 1080, // width/height of window
                SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL );
        if (window == nullptr) {
            STRATUS_ERROR << "Failed to create sdl window" << std::endl;
            SDL_Quit();
            return -1;
        }

        renderer = std::unique_ptr<stratus::Renderer>(new stratus::Renderer(window));
        if (!renderer->valid()) {
            SDL_DestroyWindow(window);
            SDL_Quit();
            return -1;
        }

        // For textures see https://3dtextures.me/
        textures.push_back(renderer->loadTexture("../resources/textures/Substance_graph_BaseColor.jpg"));
        textures.push_back(renderer->loadTexture("../resources/textures/Bark_06_basecolor.jpg"));
        textures.push_back(renderer->loadTexture("../resources/textures/Wood_Wall_003_basecolor.jpg"));
        textures.push_back(renderer->loadTexture("../resources/textures/Rock_Moss_001_basecolor.jpg"));

        normalMaps.push_back(renderer->loadTexture("../resources/textures/Substance_graph_Normal.jpg"));
        normalMaps.push_back(renderer->loadTexture("../resources/textures/Bark_06_normal.jpg"));
        normalMaps.push_back(renderer->loadTexture("../resources/textures/Wood_Wall_003_normal.jpg"));
        normalMaps.push_back(renderer->loadTexture("../resources/textures/Rock_Moss_001_normal.jpg"));

        depthMaps.push_back(renderer->loadTexture("../resources/textures/Substance_graph_Height.png"));
        depthMaps.push_back(renderer->loadTexture("../resources/textures/Bark_06_height.png"));
        depthMaps.push_back(renderer->loadTexture("../resources/textures/Wood_Wall_003_height.png"));
        depthMaps.push_back(renderer->loadTexture("../resources/textures/Rock_Moss_001_height.png"));

        roughnessMaps.push_back(renderer->loadTexture("../resources/textures/Substance_graph_Roughness.jpg"));
        roughnessMaps.push_back(renderer->loadTexture("../resources/textures/Bark_06_roughness.jpg"));
        roughnessMaps.push_back(renderer->loadTexture("../resources/textures/Wood_Wall_003_roughness.jpg"));
        roughnessMaps.push_back(renderer->loadTexture("../resources/textures/Rock_Moss_001_roughness.jpg"));

        environmentMaps.push_back(renderer->loadTexture("../resources/textures/Substance_graph_AmbientOcclusion.jpg"));
        environmentMaps.push_back(renderer->loadTexture("../resources/textures/Bark_06_ambientOcclusion.jpg"));
        environmentMaps.push_back(renderer->loadTexture("../resources/textures/Wood_Wall_003_ambientOcclusion.jpg"));
        environmentMaps.push_back(renderer->loadTexture("../resources/textures/Rock_Moss_001_ambientOcclusion.jpg"));

        outhouse = renderer->loadModel("../resources/models/Latrine.fbx");
        clay = renderer->loadModel("../resources/models/hromada_hlina_01_30k_f.FBX");
        stump = renderer->loadModel("../resources/models/boubin_stump.FBX");
        hall = renderer->loadModel("../local/hintze-hall-1m.obj");
        ramparts = renderer->loadModel("../local/model.obj");
        rocks = renderer->loadModel("../local/Rock_Terrain_SF.obj");

        for (size_t texIndex = 0; texIndex < textures.size(); ++texIndex) {
            auto cube = std::make_shared<stratus::Cube>();
            auto quad = std::make_shared<stratus::Quad>();
            stratus::RenderMaterial mat;
            mat.texture = textures[texIndex];
            mat.normalMap = normalMaps[texIndex];
            mat.depthMap = depthMaps[texIndex];
            mat.roughnessMap = roughnessMaps[texIndex];
            mat.ambientMap = environmentMaps[texIndex];
            cube->setMaterial(mat);
            quad->setMaterial(mat);
            cubeMeshes.push_back(cube);
            quadMeshes.push_back(quad);
        }

        //quadMat.texture = renderer->loadTexture("../resources/textures/volcanic_rock_texture.png");
        srand(time(nullptr));
        for (int i = 0; i < 100; ++i) {
            size_t texIndex = rand() % textures.size();
            auto mesh = quadMeshes[texIndex];
            std::unique_ptr<stratus::RenderEntity> q = std::make_unique<stratus::RenderEntity>(stratus::LightProperties::DYNAMIC);
            q->meshes.push_back(mesh);
            q->position.x = rand() % 50;
            q->position.y = rand() % 50;
            q->position.z = rand() % 50;
            q->scale = glm::vec3(float(rand() % 5));
            entities.push_back(std::move(q));
            textureIndices.push_back(texIndex);
        }
        //std::vector<std::unique_ptr<Cube>> cubes;
        stratus::RenderMaterial cubeMat;
        //cubeMat.texture = renderer->loadTexture("../resources/textures/wood_texture.jpg");
        for (int i = 0; i < 5000; ++i) {
            size_t texIndex = rand() % textures.size();
            auto mesh = cubeMeshes[texIndex];
            std::unique_ptr<stratus::RenderEntity> c = std::make_unique<stratus::RenderEntity>(stratus::LightProperties::DYNAMIC);
            c->meshes.push_back(mesh);
            c->position.x = rand() % 3000;
            c->position.y = rand() % 50;
            c->position.z = rand() % 3000;
            c->scale = glm::vec3(float(rand() % 25));
            entities.push_back(std::move(c));
            textureIndices.push_back(texIndex);
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

        cameraLight.setCastsShadows(false);
        cameraLight.setIntensity(1200.0f);
    }

    // Run a single update for the application (no infinite loops)
    // deltaSeconds = time since last frame
    virtual stratus::SystemStatus Update(double deltaSeconds) override {
        float value = 1.0f;
        if (stratus::Engine::Instance()->FrameCount() % 30 == 0) {
            STRATUS_LOG << "FPS:" << (1.0 / deltaSeconds) << std::endl;
        }
        SDL_Event e;
        const float camSpeed = 100.0f;

        // worldLight.setRotation(glm::vec3(75.0f, 0.0f, 0.0f));
        //worldLight.setRotation(stratus::Rotation(stratus::Degrees(30.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f)));
        worldLight.offsetRotation(glm::vec3(value * deltaSeconds, 0.0f, 0.0f));
        renderer->setWorldLight(worldLight);

        //STRATUS_LOG << "Camera " << camera.getYaw() << " " << camera.getPitch() << std::endl;

        // Check for key/mouse events
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_QUIT:
                    return stratus::SystemStatus::SYSTEM_SHUTDOWN;
                case SDL_MOUSEMOTION:
                    camera.modifyAngle(stratus::Degrees(0.0f), stratus::Degrees(-e.motion.xrel), stratus::Degrees(0.0f));
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
                            }
                            break;
                        case SDL_SCANCODE_R:
                            if (released) {
                                renderer->recompileShaders();
                            }
                            break;
                        case SDL_SCANCODE_I:
                            if (released) {
                                worldLightEnabled = !worldLightEnabled;
                                renderer->toggleWorldLighting(worldLightEnabled);
                                // worldLight.setColor(glm::vec3(1.0f, 0.75f, 0.5));
                                // worldLight.setColor(glm::vec3(1.0f, 0.75f, 0.75f));
                                worldLight.setColor(glm::vec3(1.0f));
                                worldLight.setIntensity(5.0f);
                                worldLight.setPosition(camera.getPosition());
                                //worldLight.setRotation(glm::vec3(90.0f, 0.0f, 0.0f));
                                renderer->setWorldLight(worldLight);
                            }
                            break;
                        case SDL_SCANCODE_1: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(1200.0f);
                                mover->light->setColor(1.0f, 1.0f, 0.5f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(1000.0);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_3: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(1500.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_4: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(2000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_5: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(3000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_6: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(6000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_7: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(12000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_8: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(24000.0f);
                                mover->light->setColor(1.0f, 0.75f, 0.5);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_9: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(48000.0f);
                                mover->light->setColor(1.0f, 0.75f, 0.5);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_0: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(65000.0f);
                                mover->light->setColor(1.0f, 1.0f, 1.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_C: {
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

        // Check mouse state
        int x, y;
        uint32_t buttonState = SDL_GetMouseState(&x, &y);
        cameraSpeed.z = 0.0f;
        if ((buttonState & SDL_BUTTON_LMASK) != 0) { // left mouse button
            cameraSpeed.z = -camSpeed;
        }
        else if ((buttonState & SDL_BUTTON_RMASK) != 0) { // right mouse button
            cameraSpeed.z = camSpeed;
        }

        // Start a new renderer frame
        renderer->begin(true);

        camera.setSpeed(cameraSpeed.y, cameraSpeed.z, cameraSpeed.x);
        camera.update(deltaSeconds);
        cameraLight.position = camera.getPosition();
        renderer->setClearColor(stratus::Color(0.0f, 0.0f, 0.0f, 1.0f));

        outhouse.scale = glm::vec3(10.0f);
        outhouse.position = glm::vec3(-50.0f, -10.0f, -45.0f);
        renderer->addDrawable(&outhouse);

        //clay.scale = glm::vec3(1.0f);
        //clay.rotation = glm::vec3(-90.0f, 0.0f, 0.0f);
        clay.position = glm::vec3(100.0f, 0.0f, -50.0f);
        clay.rotation = stratus::Rotation(stratus::Degrees(-90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f));
        renderer->addDrawable(&clay);

        stump.rotation = stratus::Rotation(stratus::Degrees(-180.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f));
        stump.position = glm::vec3(0.0f, -15.0f, -20.0f);
        renderer->addDrawable(&stump);

        hall.rotation = stratus::Rotation(stratus::Degrees(-90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f));
        hall.scale = glm::vec3(10.0f, 10.0f, 10.0f);
        hall.position = glm::vec3(-250.0f, -30.0f, 0.0f);
        renderer->addDrawable(&hall);

        ramparts.position = glm::vec3(300.0f, 0.0f, -100.0f);
        ramparts.rotation = stratus::Rotation(stratus::Degrees(90.0f), stratus::Degrees(0.0f), stratus::Degrees(0.0f));
        ramparts.scale = glm::vec3(10.0f);
        renderer->addDrawable(&ramparts);

        rocks.position = glm::vec3(700.0f, -75.0f, -100.0f);
        rocks.scale = glm::vec3(15.0f);
        renderer->addDrawable(&rocks);

        // Add the camera's light
        if (camLightEnabled) renderer->addPointLight(&cameraLight);
        for (auto & entity : entities) {
            renderer->addDrawable(entity.get());
        }

        for (auto & mover : lightMovers) {
            mover->update(deltaSeconds);
            mover->addToScene(*renderer.get());
        }
        renderer->end(camera);

        // Swap front and back buffer
        SDL_GL_SwapWindow(window);

        return stratus::SystemStatus::SYSTEM_CONTINUE;
    }

    // Perform any resource cleanup
    virtual void ShutDown() override {
        STRATUS_LOG << "Cleaning up SDL resources" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        STRATUS_LOG << "SDL resources successfully cleaned up" << std::endl;
    }

private:
    SDL_Window * window;
    std::unique_ptr<stratus::Renderer> renderer;
    std::vector<stratus::TextureHandle> textures;
    std::vector<stratus::TextureHandle> normalMaps;
    std::vector<stratus::TextureHandle> depthMaps;
    std::vector<stratus::TextureHandle> roughnessMaps;
    std::vector<stratus::TextureHandle> environmentMaps;
    stratus::Model outhouse;
    stratus::Model clay;
    stratus::Model stump;
    stratus::Model hall;
    stratus::Model ramparts;
    stratus::Model rocks;
    std::vector<std::shared_ptr<stratus::Cube>> cubeMeshes;
    std::vector<std::shared_ptr<stratus::Quad>> quadMeshes;
    std::vector<std::unique_ptr<stratus::RenderEntity>> entities;
    std::vector<size_t> textureIndices;
    glm::mat4 persp;
    stratus::Camera camera;
    glm::vec3 cameraSpeed;
    stratus::PointLight cameraLight;
    stratus::InfiniteLight worldLight;
    std::vector<std::unique_ptr<RandomLightMover>> lightMovers;
    bool camLightEnabled = true;
    bool worldLightEnabled = false;
};

STRATUS_ENTRY_POINT(StratusGFX)