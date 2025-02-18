find_package(OpenGL REQUIRED)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../)
set(THIRD_PARTY_ROOT_DIRECTORY ${ROOT_DIRECTORY}/ThirdParty/)
set(OUTPUT_NAME StratusEngine)

include_directories(${ROOT_DIRECTORY}/gl3w/include ${ROOT_DIRECTORY} ${ROOT_DIRECTORY}/Source/Engine/ . ${OPENGL_INCLUDE_DIRS})

set(SOURCES
    ${ROOT_DIRECTORY}/gl3w/src/gl3w.c
    ${CMAKE_CURRENT_LIST_DIR}/StratusCamera.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusFilesystem.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusWindow.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusEngine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusResourceManager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusEntity.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusEntityManager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusGpuMaterialBuffer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusMath.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusLog.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusGpuCommandBuffer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusTaskSystem.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusRenderComponents.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusApplicationThread.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusRendererFrontend.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusApplication.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusThread.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusTransformComponent.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusRendererBackend.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusFrameBuffer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusGpuBuffer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusTexture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusGraphicsDriver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusMaterial.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/StratusPipeline.cpp
 "StratusTypes.h")

add_library(${OUTPUT_NAME} STATIC ${SOURCES} "StratusTypes.h")

# set(OUTPUT_DIRECTORY ${ROOT_DIRECTORY}/Bin)
# set_target_properties(${OUTPUT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
# set_target_properties(${OUTPUT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
# set_target_properties(${OUTPUT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})

set(LIBRARIES ${OPENGL_LIBRARIES})

if (WIN32)
    message("Windows")
endif()

if (NOT WIN32)
    message("Linux")

    target_include_directories(
        ${OUTPUT_NAME} PUBLIC 
        ${THIRD_PARTY_ROOT_DIRECTORY}
        ${THIRD_PARTY_ROOT_DIRECTORY}/include
        ${THIRD_PARTY_ROOT_DIRECTORY}/include/SDL2
    )

    #file(GLOB LINUX_LIBS1 ${THIRD_PARTY_ROOT_DIRECTORY}/lib/*.so*)
    #file(GLOB LINUX_LIBS1 ${THIRD_PARTY_ROOT_DIRECTORY}/lib/*.a*)
    #file(GLOB LINUX_LIBS2 ${THIRD_PARTY_ROOT_DIRECTORY}/lib64/*.a*)

    list(APPEND LIBRARIES
        ${LINUX_LIBS1}
        ${LINUX_LIBS2}
        pthread
        dl
        z)
endif()

find_package(SDL2 REQUIRED)

# See https://stackoverflow.com/questions/49082471/linking-assimp-library-with-find-package
find_package(assimp REQUIRED)
if (assimp_FOUND)
    set(${ASSIMP_LIBRARY} "assimp")
endif(assimp_FOUND)

find_package(meshoptimizer REQUIRED)

target_include_directories(${OUTPUT_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})

list(APPEND LIBRARIES
    ${SDL2_LIBRARIES}
    ${ASSIMP_LIBRARIES}
    meshoptimizer::meshoptimizer
)

message(${LIBRARIES})

target_link_libraries(${OUTPUT_NAME} PUBLIC ${LIBRARIES})

install(TARGETS ${OUTPUT_NAME}
    ARCHIVE DESTINATION Bin
    LIBRARY DESTINATION Bin
    RUNTIME DESTINATION Bin)
