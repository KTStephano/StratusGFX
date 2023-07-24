set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)

set(LIBRARIES
    StratusEngine)

find_package(Catch2)

list(APPEND LIBRARIES Catch2::Catch2)

message(${LIBRARIES})

include(${ROOT_DIRECTORY}ThirdParty/lib/cmake/Catch2/Catch.cmake)

add_subdirectory(UnitTests)
add_subdirectory(IntegrationTests)
