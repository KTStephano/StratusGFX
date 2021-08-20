#include "Common.h"
#include "glm/glm.hpp"
#include <iostream>
#include <Pipeline.h>
#include <Renderer.h>
#include <Quad.h>
#include <Camera.h>
#include <chrono>
#include <Cube.h>
#include <Light.h>
#include <Utils.h>
#include <memory>

class RandomLightMover : public stratus::Entity {
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

    void update(double deltaSeconds) override {
        position = position + speed * _direction * float(deltaSeconds);
        cube->position = position;
        light->position = position;
        stratus::RenderMaterial m = cube->meshes[0]->getMaterial();
        m.diffuseColor = light->getColor() * light->getIntensity();
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
        m.diffuseColor = light->getColor() * light->getIntensity();
        cube->meshes[0]->setMaterial(m);
    }
};

int main(int argc, char * args[]) {
    auto start = std::chrono::system_clock::now();

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "Unable to initialize sdl2" << std::endl;
        return -1;
    }

    SDL_Window * window = SDL_CreateWindow("StratusGFX",
            100, 100, // location x/y on screen
            1920, 1080, // width/height of window
            SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL );
    if (window == nullptr) {
        std::cout << "Failed to create sdl window" << std::endl;
        SDL_Quit();
        return -1;
    }

    stratus::Renderer renderer(window);
    if (!renderer.valid()) {
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // For textures see https://3dtextures.me/
    std::vector<stratus::TextureHandle> textures;
    textures.push_back(renderer.loadTexture("../resources/textures/Substance_graph_BaseColor.jpg"));
    textures.push_back(renderer.loadTexture("../resources/textures/Bark_06_basecolor.jpg"));
    textures.push_back(renderer.loadTexture("../resources/textures/Wood_Wall_003_basecolor.jpg"));
    textures.push_back(renderer.loadTexture("../resources/textures/Rock_Moss_001_basecolor.jpg"));
    /**
    textures.resize(6);
    textures[0] = renderer.loadTexture("../copyrighted/brick-plaster-01-cm-big-talos.png");
    textures[1] = renderer.loadTexture("../copyrighted/brick-plaster-02-cm-big-talos.png");
    textures[2] = renderer.loadTexture("../copyrighted/cliff-01-cm-big-talos.png");
    textures[3] = renderer.loadTexture("../copyrighted/concretebare-04-cm-big-talos.png");
    textures[4] = renderer.loadTexture("../copyrighted/concreteceiling-02-cm-big-talos.png");
    textures[5] = renderer.loadTexture("../copyrighted/ruined-wall-big-talos.png");
    */

    std::vector<stratus::TextureHandle> normalMaps;
    normalMaps.push_back(renderer.loadTexture("../resources/textures/Substance_graph_Normal.jpg"));
    normalMaps.push_back(renderer.loadTexture("../resources/textures/Bark_06_normal.jpg"));
    normalMaps.push_back(renderer.loadTexture("../resources/textures/Wood_Wall_003_normal.jpg"));
    normalMaps.push_back(renderer.loadTexture("../resources/textures/Rock_Moss_001_normal.jpg"));

    /**
    normalMaps.resize(6);
    normalMaps[0] = renderer.loadTexture("../copyrighted/brick-plaster-01-nm-big-talos.png");
    normalMaps[1] = renderer.loadTexture("../copyrighted/brick-plaster-02-nm-big-talos.png");
    normalMaps[2] = renderer.loadTexture("../copyrighted/cliff-01-nm-big-talos.png");
    normalMaps[3] = renderer.loadTexture("../copyrighted/concretebare-04-nm-big-talos.png");
    normalMaps[4] = renderer.loadTexture("../copyrighted/concreteceiling-02-nm-big-talos.png");
    normalMaps[5] = renderer.loadTexture("../copyrighted/ruined-wall-nm-big-talos.png");
    */

    std::vector<stratus::TextureHandle> depthMaps;
    depthMaps.push_back(renderer.loadTexture("../resources/textures/Substance_graph_Height.png"));
    depthMaps.push_back(renderer.loadTexture("../resources/textures/Bark_06_height.png"));
    depthMaps.push_back(renderer.loadTexture("../resources/textures/Wood_Wall_003_height.png"));
    depthMaps.push_back(renderer.loadTexture("../resources/textures/Rock_Moss_001_height.png"));

    std::vector<stratus::TextureHandle> roughnessMaps;
    roughnessMaps.push_back(renderer.loadTexture("../resources/textures/Substance_graph_Roughness.jpg"));
    roughnessMaps.push_back(renderer.loadTexture("../resources/textures/Bark_06_roughness.jpg"));
    roughnessMaps.push_back(renderer.loadTexture("../resources/textures/Wood_Wall_003_roughness.jpg"));
    roughnessMaps.push_back(renderer.loadTexture("../resources/textures/Rock_Moss_001_roughness.jpg"));

    std::vector<stratus::TextureHandle> environmentMaps;
    environmentMaps.push_back(renderer.loadTexture("../resources/textures/Substance_graph_AmbientOcclusion.jpg"));
    environmentMaps.push_back(renderer.loadTexture("../resources/textures/Bark_06_ambientOcclusion.jpg"));
    environmentMaps.push_back(renderer.loadTexture("../resources/textures/Wood_Wall_003_ambientOcclusion.jpg"));
    environmentMaps.push_back(renderer.loadTexture("../resources/textures/Rock_Moss_001_ambientOcclusion.jpg"));

    stratus::Model outhouse = renderer.loadModel("../resources/models/Latrine.fbx");
    stratus::Model clay = renderer.loadModel("../resources/models/hromada_hlina_01_30k_f.FBX");
    stratus::Model stump = renderer.loadModel("../resources/models/boubin_stump.FBX");
    stratus::Model hall = renderer.loadModel("../local/hintze-hall-1m.obj");
    stratus::Model ramparts = renderer.loadModel("../local/model.obj");

    std::vector<std::shared_ptr<stratus::Cube>> cubeMeshes;
    std::vector<std::shared_ptr<stratus::Quad>> quadMeshes;
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

    std::vector<std::unique_ptr<stratus::RenderEntity>> entities;
    std::vector<size_t> textureIndices;
    stratus::RenderMaterial quadMat;
    //quadMat.texture = renderer.loadTexture("../resources/textures/volcanic_rock_texture.png");
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
    //cubeMat.texture = renderer.loadTexture("../resources/textures/wood_texture.jpg");
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
    std::vector<std::unique_ptr<RandomLightMover>> lightMovers;
    for (int x = 0; x < 3000; x += 150) {
        for (int y = 0; y < 3000; y += 150) {
            std::unique_ptr<RandomLightMover> mover(new StationaryLight());
            mover->light->setIntensity(500.0f);
            mover->position = glm::vec3(float(x),
                                        0.0f, // float(rand() % 200),
                                        float(y));
            lightMovers.push_back(std::move(mover));
        }
    }
    // for (int i = 0; i < 128; ++i) {
    //     /*
    //     std::unique_ptr<RandomLightMover> mover =
    //             std::make_unique<RandomLightMover>();
    //     mover->light->setIntensity(2500.0f);
    //     mover->position = glm::vec3(float(rand() % 3000 + 100),
    //                                 0.0f, // float(rand() % 200),
    //                                 float(rand() % 3000 + 100));
    //     lightMovers.push_back(std::move(mover));
    //     */
    //     std::unique_ptr<RandomLightMover> mover(new StationaryLight());
    //     mover->light->setIntensity(1000.0f);
    //     mover->position = glm::vec3(float(rand() % 3000 + 100),
    //                                 0.0f, // float(rand() % 200),
    //                                 float(rand() % 3000 + 100));
    //     lightMovers.push_back(std::move(mover));
    // }


    glm::mat4 persp = glm::perspective(glm::radians(90.0f), 640 / 480.0f, 0.25f, 1000.0f);

    //std::unique_ptr<Light> cameraLight(new PointLight());
    stratus::Camera camera;
    glm::vec3 cameraSpeed(0.0f);

    bool running = true;
    stratus::PointLight cameraLight;
    cameraLight.setCastsShadows(false);
    cameraLight.setIntensity(1200.0f);
    bool camLightEnabled = true;
    size_t frameCount = 0;
    float angle = 0.0f;
    while (running) {
        auto curr = std::chrono::system_clock::now();
        auto elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(curr - start).count();
        double deltaSeconds = elapsedMS / 1000.0;
        ++frameCount;
        if (frameCount % 30 == 0) std::cout << "FPS:" << (1.0 / deltaSeconds) << std::endl;
        start = curr;
        SDL_Event e;
        const float camSpeed = 100.0f;

        // Check for key/mouse events
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_MOUSEMOTION:
                    camera.modifyAngle(e.motion.xrel, 0);
                    break;
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    bool released = e.type == SDL_KEYUP;
                    SDL_Scancode key = e.key.keysym.scancode;
                    switch (key) {
                        case SDL_SCANCODE_ESCAPE:
                            if (released) running = false;
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
                                cameraSpeed.y = key == SDL_SCANCODE_D ? -camSpeed : camSpeed;
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
                                renderer.recompileShaders();
                            }
                            break;
                        case SDL_SCANCODE_1: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(250.0f);
                                mover->light->setColor(1.0f, 1.0f, 0.5f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_2: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(1000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_3: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(2500.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_4: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(5000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_5: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(10000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_6: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(20000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_7: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(40000.0f);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_8: {
                            if (released) {
                                std::unique_ptr<RandomLightMover> mover(new StationaryLight());
                                mover->light->setIntensity(65000.0f);
                                mover->light->setColor(1.0f, 0.75f, 0.5);
                                mover->position = camera.getPosition();
                                lightMovers.push_back(std::move(mover));
                            }
                            break;
                        }
                        case SDL_SCANCODE_C: {
                            lightMovers.clear();
                            break;
                        }
                        /*
                        case SDL_SCANCODE_N: {
                            if (released) {
                                stratus::RenderMaterial m;
                                for (size_t i = 0; i < entities.size(); ++i) {
                                    m = entities[i]->getMaterial();
                                    m.normalMap = m.normalMap == -1 ? normalMaps[textureIndices[i]] : -1;
                                    entities[i]->setMaterial(m);
                                }
                            }
                            break;
                        }
                        case SDL_SCANCODE_H: {
                            if (released) {
                                stratus::RenderMaterial m;
                                for (size_t i = 0; i < entities.size(); ++i) {
                                    m = entities[i]->getMaterial();
                                    m.depthMap = m.depthMap == -1 ? depthMaps[textureIndices[i]] : -1;
                                    entities[i]->setMaterial(m);
                                }
                            }
                            break;
                        }
                        */
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
        renderer.begin(true);

        angle += 10 * deltaSeconds;
        //std::cout << angle << std::endl;

        camera.setSpeed(cameraSpeed.x, cameraSpeed.z, cameraSpeed.y);
        camera.update(deltaSeconds);
        cameraLight.position = camera.getPosition();
        renderer.setClearColor(stratus::Color(0.0f, 0.0f, 0.0f, 1.0f));

        outhouse.scale = glm::vec3(10.0f);
        outhouse.position = glm::vec3(-50.0f, -10.0f, -45.0f);
        renderer.addDrawable(&outhouse);

        //clay.scale = glm::vec3(1.0f);
        //clay.rotation = glm::vec3(-90.0f, 0.0f, 0.0f);
        clay.position = glm::vec3(100.0f, 0.0f, -50.0f);
        renderer.addDrawable(&clay);

        stump.rotation = glm::vec3(-180.0f, 0.0f, 0.0f);
        stump.position = glm::vec3(0.0f, -15.0f, -20.0f);
        renderer.addDrawable(&stump);

        hall.rotation = glm::vec3(-90.0f, 0.0f, 0.0f);
        hall.scale = glm::vec3(10.0f, 10.0f, 10.0f);
        hall.position = glm::vec3(-250.0f, -30.0f, 0.0f);
        renderer.addDrawable(&hall);

        ramparts.position = glm::vec3(300.0f, 0.0f, -100.0f);
        ramparts.rotation = glm::vec3(90.0f, 0.0f, 0.0f);
        ramparts.scale = glm::vec3(10.0f);
        renderer.addDrawable(&ramparts);

        // Add the camera's light
        if (camLightEnabled) renderer.addPointLight(&cameraLight);
        for (auto & entity : entities) {
            renderer.addDrawable(entity.get());
        }

        for (auto & mover : lightMovers) {
           mover->update(deltaSeconds);
           mover->addToScene(renderer);
        }
        renderer.end(camera);

        // Swap front and back buffer
        SDL_GL_SwapWindow(window);

        /*
        glEnable(GL_BLEND);
        glEnable(GL_CULL_FACE);
        glFrontFace(GL_CW);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_POLYGON_SMOOTH);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        auto & view = camera.getViewTransform();
        glm::mat4 model(1.0f);
        quad->position.x = 15.0f;
        quad->position.z = 2.5f;
        quad->rotation.y = quad->rotation.y + (float)deltaSeconds * 10.0f;
        rotate(model, quad->rotation);
        translate(model, quad->position);
        glm::mat4 modelView = view * model;

        shader.bind();
        shader.setMat4("projection", &persp[0][0]);
        shader.setMat4("modelView", &modelView[0][0]);
        shader.setVec3("diffuseColor", &quad->getMaterial().diffuseColor[0]);
        quad->render();
        shader.unbind();
         */

        //renderer.begin(true);
        //quad->position.x = 15.0f;
        //quad->position.z = 2.5f;
        //quad->rotation.y = quad->rotation.y + (float)deltaSeconds * 10.0f;
        //renderer.addDrawable(quad);
        //Cube q;
        //Quad g;
        /*
        Quad q;
        q.position.x = 15.0f;
        q.position.z = 2.5f;
        renderer.addDrawable(&q);
         */
        /*
        shader.bind();
        for (Cube & c : cubes) {
            auto & view = camera.getViewTransform();
            glm::mat4 model(1.0f);
            rotate(model, c.rotation);
            scale(model, c.scale);
            translate(model, c.position);
            glm::mat4 modelView = view * model;
            shader.setMat4("projection", &persp[0][0]);
            shader.setMat4("modelView", &modelView[0][0]);
            glActiveTexture(GL_TEXTURE0);
            shader.setInt("diffuseTexture", 0);
            glBindTexture(GL_TEXTURE_2D, renderer._lookupTexture(c.getMaterial().texture));
            c.render();
            //glBindTexture(GL_TEXTURE_2D, 0);
        }
         */
        //renderer.end(camera);
    }

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}