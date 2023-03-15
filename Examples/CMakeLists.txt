find_package(OpenGL REQUIRED)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)

include_directories(
	${ROOT_DIRECTORY}/gl3w/include 
	${ROOT_DIRECTORY} 
	${ROOT_DIRECTORY}/Source/Engine/ 
	${CMAKE_CURRENT_LIST_DIR} 
	${OPENGL_INCLUDE_DIRS} 
	${ROOT_DIRECTORY}/assimp/Deploy/include 
	${ROOT_DIRECTORY}/SDL2/include
)

link_directories(${ROOT_DIRECTORY}/Bin)

include_directories(${CMAKE_CURRENT_LIST_DIR}/Common)
file(GLOB COMMON_SOURCES ${CMAKE_CURRENT_LIST_DIR}/Common/*.cpp)

set(OUTPUT_DIRECTORY ${ROOT_DIRECTORY}/Bin/)

add_subdirectory(ExampleEnv00)
add_subdirectory(ExampleEnv01)
add_subdirectory(ExampleEnv02)
add_subdirectory(ExampleEnv03)
add_subdirectory(ExampleEnv04)
add_subdirectory(ExampleEnv05)
add_subdirectory(ExampleEnv06)
add_subdirectory(ExampleEnv07)
add_subdirectory(ExampleEnv08)