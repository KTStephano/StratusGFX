cmake_minimum_required(VERSION 3.0)
project(StratusGFX)

set(CMAKE_CXX_STANDARD 17)
#set(SDL2_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/SDL2")
# Support both 32 and 64 bit builds

find_package(OpenGL REQUIRED)

#add_subdirectory(gl3w)

include_directories(gl3w/include includes ${SDL2_INCLUDE_DIRS} . ${OPENGL_INCLUDE_DIRS} assimp/Deploy/include)
link_directories(assimp/Deploy/lib)
file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp ${CMAKE_CURRENT_LIST_DIR}/gl3w/src/gl3w.c)
add_executable(StratusGFX ${SOURCES})

# add_executable(StratusGFX 
#     src/StratusGFX.cpp 
#     gl3w/src/gl3w.c 
#     includes/Shader.h 
#     src/Shader.cpp 
#     includes/Filesystem.h 
#     src/Filesystem.cpp 
#     includes/Common.h 
#     includes/Renderer.h 
#     src/Renderer.cpp 
#     includes/Vertex.h 
#     includes/Quad.h 
#     includes/RenderEntity.h 
#     src/RenderEntity.cpp 
#     src/Quad.cpp 
#     includes/Engine.h 
#     includes/Entity.h 
#     includes/Camera.h 
#     src/Camera.cpp 
#     includes/STBImage.h 
#     includes/Cube.h 
#     src/Cube.cpp 
#     includes/Light.h 
#     includes/Utils.h)

set(OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin/)
set_target_properties(StratusGFX PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
if (WIN32)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/SDL2/include)
    if (${CMAKE_SIZEOF_VOID_P} MATCHES 8)
        #set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2main.lib")
        file(GLOB SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/SDL2/lib/x64/*.lib")
        set(SDL2 "${CMAKE_CURRENT_LIST_DIR}/SDL2/lib/x64/SDL2.dll")
    else ()
        file(GLOB SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/SDL2/lib/x86/*.lib")
        set(SDL2 "${CMAKE_CURRENT_LIST_DIR}/SDL2/lib/x86/SDL2.dll")
        #set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x86/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x86/SDL2main.lib")
    endif ()
    message(${SDL2})
    file(COPY ${SDL2} DESTINATION ${OUTPUT_DIRECTORY})
    #file(COPY "assimp/Deploy/lib/assimp-vc142-mt.dll" DESTINATION ${OUTPUT_DIRECTORY})
    file(GLOB ASSIMP_LIBS "${CMAKE_CURRENT_LIST_DIR}/assimp/Deploy/lib/*.lib")
    file(GLOB ASSIMP_DLLS "${CMAKE_CURRENT_LIST_DIR}/assimp/Deploy/lib/*.dll")
    file(COPY ${ASSIMP_DLLS} DESTINATION ${OUTPUT_DIRECTORY})
    target_link_libraries(StratusGFX ${SDL2_LIBRARIES} opengl32 kernel32 user32 gdi32 winspool comdlg32 ${ASSIMP_LIBS})
else ()
    find_package(SDL2 REQUIRED)
    #target_link_libraries(StratusGFX ${SDL2_LIBRARIES} ${OPENGL_LIBRARIES} GLU X11 Xxf86vm Xrandr pthread Xi dl)
    target_link_libraries(StratusGFX ${SDL2_LIBRARIES} ${OPENGL_LIBRARIES} pthread dl assimp)
endif ()
