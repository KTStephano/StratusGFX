#include "includes/Common.h"
#include "glm/glm.hpp"
#include <iostream>
#include <includes/Shader.h>
#include <includes/Renderer.h>
#include <includes/Quad.h>
#include <includes/Camera.h>
#include <chrono>
#include <includes/Cube.h>
#include <includes/Light.h>
#include <includes/Utils.h>

static const std::vector<GLfloat> cubeData = std::vector<GLfloat>{
    -1.0f, 1.0f, 0.0f,  0, 0, 0,    0.0f, 1.0f,
    -1.0f, -1.0f, 0.0f, 0, 0, 0,    0.0f, 0.0f,
    1.0f, -1.0f, 0.0f,  0, 0, 0,    1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f,  0, 0, 0,    0.0f, 1.0f,
    1.0f, -1.0f, 0.0f,  0, 0, 0,    1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,   0, 0, 0,    1.0f, 1.0f
};

static std::vector<glm::vec3> genCubePositions() {
    std::vector<glm::vec3> pos;
    for (size_t index = 0; index < cubeData.size(); index += 8) {
        glm::vec3 p(cubeData[index], cubeData[index + 1], cubeData[index + 2]);
        pos.push_back(p);
    }
    return pos;
}

static std::vector<glm::vec2> genCubeTexCoords() {
    std::vector<glm::vec2> coords;
    for (size_t index = 6; index < cubeData.size(); index += 8) {
        coords.push_back(glm::vec2(cubeData[index], cubeData[index + 1]));
    }
    return coords;
}

static void calcTangents() {
    auto positions = genCubePositions();
    auto coords = genCubeTexCoords();
    for (size_t i = 0; i < positions.size(); i += 3) {
        glm::vec3 p1 = positions[i];
        glm::vec3 p2 = positions[i + 1];
        glm::vec3 p3 = positions[i + 2];

        glm::vec2 uv1 = coords[i];
        glm::vec2 uv2 = coords[i + 1];
        glm::vec2 uv3 = coords[i + 2];

        auto tanBitan = calculateTangentAndBitangent(p1, p2, p3,
                                                     uv1, uv2, uv3);
        auto tangent = tanBitan.first;
        auto bitangent = tanBitan.second;
        std::cout << tangent.x << ", " << tangent.y << ", " << tangent.z << ", "
            << bitangent.x << ", " << bitangent.y << ", " << bitangent.z << ", " << std::endl;
    }
}

class RandomLightMover : public Entity {
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
    std::unique_ptr<Cube> cube;
    std::unique_ptr<Light> light;

    RandomLightMover() {
        cube = std::make_unique<Cube>();
        light = std::make_unique<PointLight>();
        speed = glm::vec3(float(rand() % 15 + 10));
        _changeDirection();
    }

    void update(double deltaSeconds) override {
        position = position + speed * _direction * float(deltaSeconds);
        cube->position = position;
        light->position = position;
        RenderMaterial m = cube->getMaterial();
        m.diffuseColor = light->getColor() * light->getIntensity();
        cube->setMaterial(m);

        _elapsedSec += deltaSeconds;
        if (_elapsedSec > 5.0) {
            _elapsedSec = 0.0;
            _changeDirection();
        }
    }
};

static void rotate(glm::mat4 & out, const glm::vec3 & angles) {
    float angleX = glm::radians(angles.x);
    float angleY = glm::radians(angles.y);
    float angleZ = glm::radians(angles.z);

    float cx = std::cos(angleX);
    float cy = std::cos(angleY);
    float cz = std::cos(angleZ);

    float sx = std::sin(angleX);
    float sy = std::sin(angleY);
    float sz = std::sin(angleZ);

    out[0] = glm::vec4(cy * cz,
                       sx * sy * cz + cx * sz,
                       -cx * sy * cz + sx * sz,
                       out[0].w);

    out[1] = glm::vec4(-cy * sz,
                       -sx * sy * sz + cx * cz,
                       cx * sy * sz + sx * cz,
                       out[1].w);

    out[2] = glm::vec4(sy,
                       -sx * cy,
                       cx * cy, out[2].w);
}

// Inserts a 3x3 matrix into the upper section of a 4x4 matrix
static void inset(glm::mat4 & out, const glm::mat3 & in) {
    out[0].x = in[0].x;
    out[0].y = in[0].y;
    out[0].z = in[0].z;

    out[1].x = in[1].x;
    out[1].y = in[1].y;
    out[1].z = in[1].z;

    out[2].x = in[2].x;
    out[2].y = in[2].y;
    out[2].z = in[2].z;
}

static void scale(glm::mat4 & out, const glm::vec3 & scale) {
    out[0].x = out[0].x * scale.x;
    out[0].y = out[0].y * scale.y;
    out[0].z = out[0].z * scale.z;

    out[1].x = out[1].x * scale.x;
    out[1].y = out[1].y * scale.y;
    out[1].z = out[1].z * scale.z;

    out[2].x = out[2].x * scale.x;
    out[2].y = out[2].y * scale.y;
    out[2].z = out[2].z * scale.z;
}

static void translate(glm::mat4 & out, const glm::vec3 & translate) {
    out[3].x = translate.x;
    out[3].y = translate.y;
    out[3].z = translate.z;
}

int main(int argc, char * args[]) {
    calcTangents();

    auto start = std::chrono::system_clock::now();

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "Unable to initialize sdl2" << std::endl;
        return -1;
    }

    SDL_Window * window = SDL_CreateWindow("StratusGFX",
            100, 100, // location x/y on screen
            1280, 720, // width/height of window
            SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL );
    if (window == nullptr) {
        std::cout << "Failed to create sdl window" << std::endl;
        SDL_Quit();
        return -1;
    }

    Renderer renderer(window);
    if (!renderer.valid()) {
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    std::cout << "Renderer: " << renderer.config().renderer << std::endl;
    std::cout << "GL version: " << renderer.config().version << std::endl;

    Shader shader("../resources/shaders/texture_no_lighting.vs",
            "../resources/shaders/texture_no_lighting.fs");
    Shader shader2("../resources/shaders/shader.vs",
            "../resources/shaders/shader.fs");
    if (!shader.isValid() || !shader2.isValid()) return -1;


    std::vector<TextureHandle> textures;
    textures.resize(6);
    textures[0] = renderer.loadTexture("../copyrighted/brick-plaster-01-cm-big-talos.png");
    textures[1] = renderer.loadTexture("../copyrighted/brick-plaster-02-cm-big-talos.png");
    textures[2] = renderer.loadTexture("../copyrighted/cliff-01-cm-big-talos.png");
    textures[3] = renderer.loadTexture("../copyrighted/concretebare-04-cm-big-talos.png");
    textures[4] = renderer.loadTexture("../copyrighted/concreteceiling-02-cm-big-talos.png");
    textures[5] = renderer.loadTexture("../copyrighted/ruined-wall-big-talos.png");

    std::vector<TextureHandle> normalMaps;
    normalMaps.resize(6);
    normalMaps[0] = renderer.loadTexture("../copyrighted/brick-plaster-01-nm-big-talos.png");
    normalMaps[1] = renderer.loadTexture("../copyrighted/brick-plaster-02-nm-big-talos.png");
    normalMaps[2] = renderer.loadTexture("../copyrighted/cliff-01-nm-big-talos.png");
    normalMaps[3] = renderer.loadTexture("../copyrighted/concretebare-04-nm-big-talos.png");
    normalMaps[4] = renderer.loadTexture("../copyrighted/concreteceiling-02-nm-big-talos.png");
    normalMaps[5] = renderer.loadTexture("../copyrighted/ruined-wall-nm-big-talos.png");

    std::vector<std::unique_ptr<RenderEntity>> entities;
    std::vector<size_t> textureIndices;
    RenderMaterial quadMat;
    //quadMat.texture = renderer.loadTexture("../resources/textures/volcanic_rock_texture.png");
    std::cout << quadMat.texture << std::endl;
    srand(time(nullptr));
    for (int i = 0; i < 100; ++i) {
        size_t texIndex = rand() % textures.size();
        quadMat.texture = textures[texIndex];
        quadMat.normalMap = normalMaps[texIndex];
        std::unique_ptr<Quad> q = std::make_unique<Quad>();
        q->setMaterial(quadMat);
        q->position.x = rand() % 50;
        q->position.y = rand() % 50;
        q->position.z = rand() % 50;
        q->scale = glm::vec3(float(rand() % 5));
        q->enableLightInteraction(true);
        entities.push_back(std::move(q));
        textureIndices.push_back(texIndex);
    }
    //std::vector<std::unique_ptr<Cube>> cubes;
    RenderMaterial cubeMat;
    //cubeMat.texture = renderer.loadTexture("../resources/textures/wood_texture.jpg");
    for (int i = 0; i < 2000; ++i) {
        std::unique_ptr<Cube> c = std::make_unique<Cube>();
        size_t texIndex = rand() % textures.size();
        cubeMat.texture = textures[texIndex];
        cubeMat.normalMap = normalMaps[texIndex];
        c->setMaterial(cubeMat);
        c->position.x = rand() % 750;
        c->position.y = rand() % 750;
        c->position.z = rand() % 750;
        c->scale = glm::vec3(float(rand() % 25));
        c->enableLightInteraction(true);
        entities.push_back(std::move(c));
        textureIndices.push_back(texIndex);
    }

    // Create the light movers
    std::vector<std::unique_ptr<RandomLightMover>> lightMovers;
    for (int i = 0; i < 3; ++i) {
        std::unique_ptr<RandomLightMover> mover =
                std::make_unique<RandomLightMover>();
        mover->light->setIntensity(200.0f);
        mover->position = glm::vec3(float(rand() % 500 + 100),
                                    0.0f, // float(rand() % 200),
                                    float(rand() % 500 + 100));
        lightMovers.push_back(std::move(mover));
    }

    glm::mat4 persp = glm::perspective(glm::radians(90.0f), 640 / 480.0f, 0.25f, 1000.0f);

    Camera camera;
    glm::vec3 cameraSpeed(0.0f);

    bool running = true;
    PointLight cameraLight;
    cameraLight.setIntensity(200.0f);
    while (running) {
        auto curr = std::chrono::system_clock::now();
        auto elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(curr - start).count();
        double deltaSeconds = elapsedMS / 1000.0;
        //std::cout << deltaSeconds << std::endl;
        start = curr;
        SDL_Event e;
        const float camSpeed = 100.0f;
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
                        case SDL_SCANCODE_N: {
                            if (released) {
                                RenderMaterial m;
                                for (size_t i = 0; i < entities.size(); ++i) {
                                    m = entities[i]->getMaterial();
                                    m.normalMap = m.normalMap == -1 ? normalMaps[textureIndices[i]] : -1;
                                    entities[i]->setMaterial(m);
                                }
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

        camera.setSpeed(cameraSpeed.x, cameraSpeed.z, cameraSpeed.y);
        camera.update(deltaSeconds);
        cameraLight.position = camera.getPosition();
        renderer.setClearColor(Color(0.0f, 0.0f, 0.0f, 1.0f));

        renderer.begin(true);
        // Add the camera's light
        //renderer.addPointLight(&cameraLight);
        for (auto & entity : entities) {
            renderer.addDrawable(entity.get());
        }
        for (auto & mover : lightMovers) {
            mover->update(deltaSeconds);
            renderer.addPointLight(mover->light.get());
            renderer.addDrawable(mover->cube.get());
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