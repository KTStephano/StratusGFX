cmake_minimum_required(VERSION 3.0)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})
project(StratusGFX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(Source/)
add_subdirectory(Examples/)

include(CTest)
enable_testing()

add_subdirectory(Tests)