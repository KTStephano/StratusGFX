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

    Shader shader("../resources/shaders/texture_no_lighting.vs",
            "../resources/shaders/texture_no_lighting.fs");
    if (!shader.isValid()) return -1;

    Quad quad(RenderMode::PERSPECTIVE);

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
                                cameraSpeed.x = key == SDL_SCANCODE_W ? 5.0f : -5.0f;
                            } else {
                                cameraSpeed.x = 0.0f;
                            }
                            break;
                        case SDL_SCANCODE_A:
                        case SDL_SCANCODE_D:
                            if (!released) {
                                cameraSpeed.y = key == SDL_SCANCODE_D ? 5.0f : -5.0f;
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

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera.setSpeed(cameraSpeed.x, cameraSpeed.z, cameraSpeed.y);
        camera.update(deltaSeconds);
        
        //glViewport(0, 0, 230, 230);
        shader.bind();
        quad.render();
        shader.unbind();

        // Swap front and back buffer
        SDL_GL_SwapWindow(window);
    }

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}