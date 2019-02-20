#include "includes/Common.h"
#include "glm/glm.hpp"
#include <iostream>
#include <includes/Shader.h>
#include <includes/Renderer.h>
#include <includes/Quad.h>
#include <includes/Camera.h>
#include <chrono>

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

    Shader shader("../resources/shaders/no_texture_no_lighting.vs",
            "../resources/shaders/no_texture_no_lighting.fs");
    Shader shader2("../resources/shaders/shader.vs",
            "../resources/shaders/shader.fs");
    if (!shader.isValid() || !shader2.isValid()) return -1;

    std::vector<Quad> quads;
    RenderMaterial quadMat;
    quadMat.texture = renderer.loadTexture("../resources/textures/volcanic_rock_texture.png");
    std::cout << quadMat.texture << std::endl;
    srand(time(nullptr));
    for (int i = 0; i < 100; ++i) {
        Quad q;
        q.setMaterial(quadMat);
        q.position.x = rand() % 50;
        q.position.y = rand() % 50;
        q.position.z = rand() % 50;
        q.scale = glm::vec3(float(rand() % 5));
        quads.push_back(q);
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

        renderer.begin(true);
        //quad->position.x = 15.0f;
        //quad->position.z = 2.5f;
        //quad->rotation.y = quad->rotation.y + (float)deltaSeconds * 10.0f;
        //renderer.addDrawable(quad);
        Quad q;
        //Quad g;
        /*
        Quad q;
        q.position.x = 15.0f;
        q.position.z = 2.5f;
        renderer.addDrawable(&q);
         */
        for (int i = 0; i < quads.size(); ++i) {
            renderer.addDrawable(&quads[i]);
        }
        renderer.end(camera);

        // Swap front and back buffer
        SDL_GL_SwapWindow(window);
    }

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}