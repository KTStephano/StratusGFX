cmake_minimum_required(VERSION 3.15)
#set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})
project(StratusGFX)

set(CMAKE_CXX_STANDARD 17)

# if (WIN32)
#     message($ENV{SYSROOT})
#     list(APPEND CMAKE_PREFIX_PATH 
#         $ENV{SYSROOT}/cmake 
#         $ENV{SYSROOT}/lib/cmake 
#         $ENV{SYSROOT}/lib/cmake/assimp-5.0 
#         $ENV{SYSROOT}/lib/cmake/Catch2
#         $ENV{SYSROOT}/lib/cmake/meshoptimizer
#     )
# endif ()

list(APPEND CMAKE_PREFIX_PATH
    ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/cmake
    ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/lib/cmake
    ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/lib/cmake/Catch2
    ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/lib/cmake/assimp
    ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/lib/cmake/assimp-5.2
    ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/lib/cmake/meshoptimizer)

option(DEPENDENCY_BUILD "Third party dependencies only" OFF)
option(BUILD_TESTS "Build engine integration and unit tests" ON)

file(GLOB BIN_DLLS ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/bin/*)
    
file(
    COPY ${BIN_DLLS}
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/Bin
)

# See https://stackoverflow.com/questions/7747857/in-cmake-how-do-i-work-around-the-debug-and-release-directories-visual-studio-2
# First for the generic no-config case (e.g. with mingw)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Bin )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Bin )
# Second, for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_CURRENT_LIST_DIR}/Bin )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_CURRENT_LIST_DIR}/Bin )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_CURRENT_LIST_DIR}/Bin )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

include_directories(${CMAKE_CURRENT_LIST_DIR}/stb)

add_subdirectory(Source/)
add_subdirectory(Examples/)

#include(CTest)
#enable_testing()

if (BUILD_TESTS)
    add_subdirectory(Tests)
endif()