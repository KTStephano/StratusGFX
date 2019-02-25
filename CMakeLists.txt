cmake_minimum_required(VERSION 3.0)
project(StratusGFX)

set(CMAKE_CXX_STANDARD 17)
#set(SDL2_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/SDL2")
# Support both 32 and 64 bit builds

include_directories(gl3w/ .)
add_executable(StratusGFX 
    src/StratusGFX.cpp 
    src/gl3w.c 
    includes/Shader.h 
    src/Shader.cpp 
    includes/Filesystem.h 
    src/Filesystem.cpp 
    includes/Common.h 
    includes/Renderer.h 
    src/Renderer.cpp 
    includes/Vertex.h 
    includes/Quad.h 
    includes/RenderEntity.h 
    src/RenderEntity.cpp 
    src/Quad.cpp 
    includes/Engine.h 
    includes/Entity.h 
    includes/Camera.h 
    src/Camera.cpp 
    includes/STBImage.h 
    includes/Cube.h 
    src/Cube.cpp 
    includes/Light.h 
    includes/Utils.h)

set_target_properties(StratusGFX PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin/)
if (WIN32)
    if (${CMAKE_SIZEOF_VOID_P} MATCHES 8)
        set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2main.lib")
    else ()
        set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x86/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x86/SDL2main.lib")
    endif ()
    target_link_libraries(StratusGFX ${SDL2_LIBRARIES} opengl32 kernel32 user32 gdi32 winspool comdlg32)
else ()
    target_link_libraries(StratusGFX SDL2 GL GLU X11 Xxf86vm Xrandr pthread Xi dl)
endif ()