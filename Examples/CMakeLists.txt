include_directories(${CMAKE_CURRENT_LIST_DIR}/Common)

set(COMMON_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/Common/LightControllers.cpp
)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)

include_directories(
    ${ROOT_DIRECTORY}gl3w/include 
    ${ROOT_DIRECTORY} 
    ${ROOT_DIRECTORY}Source/Engine/ 
    ${CMAKE_CURRENT_LIST_DIR} 
    ${CMAKE_CURRENT_LIST_DIR}/Common
    ${OPENGL_INCLUDE_DIRS}
)

link_directories(${ROOT_DIRECTORY}Bin)

set(LIBRARIES StratusEngine)

add_subdirectory(ExampleEnv00)
add_subdirectory(ExampleEnv01)
add_subdirectory(ExampleEnv02)
add_subdirectory(ExampleEnv03)
add_subdirectory(ExampleEnv04)
add_subdirectory(ExampleEnv05)
add_subdirectory(ExampleEnv06)
add_subdirectory(ExampleEnv07)
add_subdirectory(ExampleEnv08)
add_subdirectory(ExampleEnv09)