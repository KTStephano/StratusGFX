cmake_minimum_required(VERSION 3.0)
#set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})
project(StratusGFX)

set(CMAKE_CXX_STANDARD 17)

if (WIN32)
    message($ENV{SYSROOT})
    list(APPEND CMAKE_PREFIX_PATH 
        $ENV{SYSROOT}/cmake 
        $ENV{SYSROOT}/lib/cmake 
        $ENV{SYSROOT}/lib/cmake/assimp-5.0 
        $ENV{SYSROOT}/lib/cmake/Catch2
        $ENV{SYSROOT}/lib/cmake/meshoptimizer
    )
endif ()

include_directories(${CMAKE_CURRENT_LIST_DIR}/stb)

add_subdirectory(Source/)
add_subdirectory(Examples/)

include(CTest)
enable_testing()

add_subdirectory(Tests)