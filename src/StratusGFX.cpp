#include "includes/Common.h"
#include "glm/glm.hpp"
#include <iostream>
#include <includes/Shader.h>
#include <includes/Renderer.h>
#include <includes/Quad.h>
#include <includes/Camera.h>
#include <chrono>
#include <includes/Cube.h>

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
    std::cout << args[0] << std::endl;

    auto start = std::chrono::system_clock::now();

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "Unable to initialize sdl2" << std::endl;
        return -1;
    }

    SDL_Window * window = SDL_CreateWindow("StratusGFX",
            100, 100, // location x/y on screen
            640, 480, // width/height of window
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

    std::vector<std::unique_ptr<RenderEntity>> entities;
    RenderMaterial quadMat;
    quadMat.texture = renderer.loadTexture("../resources/textures/volcanic_rock_texture.png");
    std::cout << quadMat.texture << std::endl;
    srand(time(nullptr));
    for (int i = 0; i < 100; ++i) {
        std::unique_ptr<Quad> q = std::make_unique<Quad>();
        q->setMaterial(quadMat);
        q->position.x = rand() % 50;
        q->position.y = rand() % 50;
        q->position.z = rand() % 50;
        q->scale = glm::vec3(float(rand() % 5));
        q->enableLightInteraction(true);
        entities.push_back(std::move(q));
    }
    //std::vector<std::unique_ptr<Cube>> cubes;
    RenderMaterial cubeMat;
    cubeMat.texture = renderer.loadTexture("../resources/textures/wood_texture.jpg");
    for (int i = 0; i < 100; ++i) {
        std::unique_ptr<Cube> c = std::make_unique<Cube>();
        c->setMaterial(cubeMat);
        c->position.x = rand() % 100;
        c->position.y = rand() % 100;
        c->position.z = rand() % 100;
        c->scale = glm::vec3(float(rand() % 10));
        c->enableLightInteraction(true);
        entities.push_back(std::move(c));
    }
    glm::mat4 persp = glm::perspective(glm::radians(90.0f), 640 / 480.0f, 0.25f, 1000.0f);

    Camera camera;
    glm::vec3 cameraSpeed(0.0f);

    bool running = true;
    while (running) {
        auto curr = std::chrono::system_clock::now();
        auto elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(curr - start).count();
        double deltaSeconds = elapsedMS / 1000.0;
        //std::cout << deltaSeconds << std::endl;
        start = curr;
        SDL_Event e;
        const float camSpeed = 50.0f;
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
                        default: break;
                    }
                    break;
                }
                default: break;
            }
        }

        camera.setSpeed(cameraSpeed.x, cameraSpeed.z, cameraSpeed.y);
        camera.update(deltaSeconds);

        renderer.begin(true);
        for (auto & entity : entities) {
            renderer.addDrawable(entity.get());
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