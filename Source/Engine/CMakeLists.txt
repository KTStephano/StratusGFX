find_package(OpenGL REQUIRED)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../)
set(THIRD_PARTY_ROOT_DIRECTORY ${ROOT_DIRECTORY}/ThirdParty/)
set(OUTPUT_NAME StratusEngine)

include_directories(${ROOT_DIRECTORY}/gl3w/include ${ROOT_DIRECTORY} ${ROOT_DIRECTORY}/Source/Engine/ . ${OPENGL_INCLUDE_DIRS})
file(GLOB SOURCES ${ROOT_DIRECTORY}/Source/Engine/*.cpp ${ROOT_DIRECTORY}/gl3w/src/gl3w.c)
add_library(${OUTPUT_NAME} STATIC ${SOURCES})

# set(OUTPUT_DIRECTORY ${ROOT_DIRECTORY}/Bin)
# set_target_properties(${OUTPUT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
# set_target_properties(${OUTPUT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
# set_target_properties(${OUTPUT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})

set(LIBRARIES ${OPENGL_LIBRARIES})

if (WIN32)
    message("Windows")
    find_package(SDL2 REQUIRED)

    # See https://stackoverflow.com/questions/49082471/linking-assimp-library-with-find-package
    find_package(assimp REQUIRED)
    if (assimp_FOUND)
        set(${ASSIMP_LIBRARY} "assimp")
    endif(assimp_FOUND)

    find_package(meshoptimizer REQUIRED)

    target_include_directories(${OUTPUT_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
    #message(${SDL2_LIBRARIES})
    #message(${ASSIMP_LIBRARIES})

    list(APPEND LIBRARIES
        ${SDL2_LIBRARIES}
        ${ASSIMP_LIBRARIES}
        meshoptimizer::meshoptimizer
    )
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
    file(GLOB LINUX_LIBS1 ${THIRD_PARTY_ROOT_DIRECTORY}/lib/*.a*)

    list(APPEND LIBRARIES
        ${LINUX_LIBS1}
        ${LINUX_LIBS2}
        pthread
        dl
        z)
endif()

message(${LIBRARIES})

target_link_libraries(${OUTPUT_NAME} PUBLIC ${LIBRARIES})

install(TARGETS ${OUTPUT_NAME}
    ARCHIVE DESTINATION Bin
    LIBRARY DESTINATION Bin
    RUNTIME DESTINATION Bin)
