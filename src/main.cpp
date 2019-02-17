#include "SDL2/SDL.h"
#include <iostream>

int main(int argc, char * args[]) {

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "Unable to initialize sdl2" << std::endl;
        return -1;
    }

    SDL_Window * window = SDL_CreateWindow("StratusGFX", 100, 100, 640, 480, SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        std::cout << "Failed to create sdl window" << std::endl;
        SDL_Quit();
        return -1;
    }

    // Note that -1 tells it to pick the first rendering driver that meets
    // out requirements
    SDL_Renderer * renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == nullptr) {
        std::cout << "Unable to create hardware accelerated renderer" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    while (true) {
        SDL_RenderClear(renderer);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}