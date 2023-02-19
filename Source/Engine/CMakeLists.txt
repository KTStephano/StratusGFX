find_package(OpenGL REQUIRED)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../)
message(${ROOT_DIRECTORY})
set(OUTPUT_NAME StratusEngine)

include_directories(${ROOT_DIRECTORY}/gl3w/include ${ROOT_DIRECTORY} ${ROOT_DIRECTORY}/Source/Engine/ . ${OPENGL_INCLUDE_DIRS})
link_directories(assimp/Deploy/lib)
file(GLOB SOURCES ${ROOT_DIRECTORY}/Source/Engine/*.cpp ${ROOT_DIRECTORY}/gl3w/src/gl3w.c)
add_library(${OUTPUT_NAME} STATIC ${SOURCES})

set(OUTPUT_DIRECTORY ${ROOT_DIRECTORY}/Bin/)
set_target_properties(${OUTPUT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
set_target_properties(${OUTPUT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
set_target_properties(${OUTPUT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})

find_package(SDL2 REQUIRED)

# See https://stackoverflow.com/questions/49082471/linking-assimp-library-with-find-package
find_package(assimp REQUIRED)
if (assimp_FOUND)
    set(${ASSIMP_LIBRARY} "assimp")
endif(assimp_FOUND)

target_include_directories(${OUTPUT_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
message(${SDL2_LIBRARIES})
message(${ASSIMP_LIBRARIES})

target_link_directories(${OUTPUT_NAME} PUBLIC ${ROOT_DIRECTORY}/assimp/Deploy/lib/)

set(LIBRARIES
    ${SDL2_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${ASSIMP_LIBRARIES}
)

if (NOT WIN32)
    message("hello")
    list(APPEND LIBRARIES
        pthread
        dl)
endif()

target_link_libraries(${OUTPUT_NAME} PUBLIC ${LIBRARIES})

install(TARGETS ${OUTPUT_NAME}
    ARCHIVE DESTINATION Bin
    LIBRARY DESTINATION Bin
    RUNTIME DESTINATION Bin)
