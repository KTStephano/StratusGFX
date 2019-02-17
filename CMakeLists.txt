cmake_minimum_required(VERSION 3.13)
project(StratusGFX)

set(CMAKE_CXX_STANDARD 17)

include_directories(gl3w/)
add_executable(StratusGFX src/main.cpp src/gl3w.c)

target_link_libraries(StratusGFX SDL2 GL GLU X11 Xxf86vm Xrandr pthread Xi dl)
