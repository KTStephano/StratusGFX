#include "includes/Common.h"
#include "glm/glm.hpp"
#include <iostream>
#include <includes/Shader.h>
#include <includes/Renderer.h>

int main(int argc, char * args[]) {
    std::cout << args[0] << std::endl;

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

    Shader shader("../resources/shaders/shader.vs", "../resources/shaders/shader.fs");
    std::cout << std::boolalpha;
    std::cout << shader.isValid() << std::endl;

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                default: break;
            }
        }

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Swap front and back buffer
        SDL_GL_SwapWindow(window);
    }

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}