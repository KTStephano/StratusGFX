set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)

set(LIBRARIES
    StratusEngine)

if (WIN32)
    find_package(Catch2)

    list(APPEND LIBRARIES Catch2::Catch2)
endif ()

if (NOT WIN32)
    file(GLOB LINUX_LIBS1 ${ROOT_DIRECTORY}ThirdParty/lib/*.a*)

    list(APPEND LIBRARIES ${LINUX_LIBS1})
endif ()

message(${LIBRARIES})

include(${ROOT_DIRECTORY}ThirdParty/lib/cmake/Catch2/Catch.cmake)

add_subdirectory(UnitTests)
add_subdirectory(IntegrationTests)
