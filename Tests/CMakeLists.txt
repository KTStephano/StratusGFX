find_package(Catch2 REQUIRED)
find_package(OpenGL REQUIRED)

set(TEST_EXE StratusEngineTests)
file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)

link_directories(${ROOT_DIRECTORY}/Bin)

add_executable(${TEST_EXE} ${TEST_SOURCES})

target_include_directories(${TEST_EXE} PUBLIC
    ${ROOT_DIRECTORY}/gl3w/include
    ${ROOT_DIRECTORY}
    ${ROOT_DIRECTORY}/Source/
    ${CMAKE_CURRENT_LIST_DIR}
    ${OPENGL_INCLUDE_DIRS}
    ${ROOT_DIRECTORY}/assimp/Deploy/include
    ${ROOT_DIRECTORY}/SDL2/include
)

target_link_libraries(${TEST_EXE}
    Catch2::Catch2
    StratusEngine
)

include(CTest)
include(Catch)
catch_discover_tests(${TEST_EXE})