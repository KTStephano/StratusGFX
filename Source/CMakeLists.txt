find_package(OpenGL REQUIRED)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)
set(OUTPUT_NAME StratusEngine)

include_directories(${ROOT_DIRECTORY}/gl3w/include ${ROOT_DIRECTORY} ${ROOT_DIRECTORY}/Source/ . ${OPENGL_INCLUDE_DIRS} ${ROOT_DIRECTORY}/assimp/Deploy/include)
link_directories(assimp/Deploy/lib)
file(GLOB SOURCES ${ROOT_DIRECTORY}/Source/*.cpp ${ROOT_DIRECTORY}/gl3w/src/gl3w.c)
add_library(${OUTPUT_NAME} STATIC ${SOURCES})

set(OUTPUT_DIRECTORY ${ROOT_DIRECTORY}/Bin/)
if (WIN32)
    include_directories(${ROOT_DIRECTORY}/SDL2/include)
    if (${CMAKE_SIZEOF_VOID_P} MATCHES 8)
        #set(SDL2_LIBRARIES "${ROOT_DIRECTORY}/lib/x64/SDL2.lib;${ROOT_DIRECTORY}/lib/x64/SDL2main.lib")
        file(GLOB SDL2_LIBRARIES "${ROOT_DIRECTORY}/SDL2/lib/x64/*.lib")
        set(SDL2 "${ROOT_DIRECTORY}/SDL2/lib/x64/SDL2.dll")
    else ()
        file(GLOB SDL2_LIBRARIES "${ROOT_DIRECTORY}/SDL2/lib/x86/*.lib")
        set(SDL2 "${ROOT_DIRECTORY}/SDL2/lib/x86/SDL2.dll")
        #set(SDL2_LIBRARIES "${ROOT_DIRECTORY}/lib/x86/SDL2.lib;${ROOT_DIRECTORY}/lib/x86/SDL2main.lib")
    endif ()
    file(COPY ${SDL2} DESTINATION ${OUTPUT_DIRECTORY})
    file(GLOB ASSIMP_LIBS "${ROOT_DIRECTORY}/assimp/Deploy/lib/*.lib")
    file(GLOB ASSIMP_DLLS "${ROOT_DIRECTORY}/assimp/Deploy/lib/*.dll")
    file(COPY ${ASSIMP_DLLS} DESTINATION ${OUTPUT_DIRECTORY})
    target_link_libraries(${OUTPUT_NAME} ${SDL2_LIBRARIES} opengl32 kernel32 user32 gdi32 winspool comdlg32 ${ASSIMP_LIBS})
else ()
    find_package(SDL2 REQUIRED)
    include_directories(${SDL2_INCLUDE_DIRS})
    target_link_libraries(${OUTPUT_NAME} ${SDL2_LIBRARIES} ${OPENGL_LIBRARIES} pthread dl assimp)
endif ()

install(TARGETS ${OUTPUT_NAME}
    ARCHIVE DESTINATION Bin
    LIBRARY DESTINATION Bin
    RUNTIME DESTINATION Bin)