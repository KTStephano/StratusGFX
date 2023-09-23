set(TEST_EXE StratusEngineUnitTests)

set(TEST_SOURCES 
    ${CMAKE_CURRENT_LIST_DIR}/ThreadTest.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TestPoolAllocators.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TestUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/HandleTest.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TestUnsafePtr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TestStackAllocators.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TestConcurrentHashMap.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TestsMain.cpp
)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../)

link_directories(${ROOT_DIRECTORY}/Bin)

add_executable(${TEST_EXE} ${TEST_SOURCES})

target_include_directories(${TEST_EXE} PUBLIC
    ${ROOT_DIRECTORY}/gl3w/include
    ${ROOT_DIRECTORY}
    ${ROOT_DIRECTORY}/Source/Engine/
    ${CMAKE_CURRENT_LIST_DIR}
    ${OPENGL_INCLUDE_DIRS}
    ${ROOT_DIRECTORY}/assimp/Deploy/include
    ${ROOT_DIRECTORY}/SDL2/include
)

target_link_libraries(${TEST_EXE}
    ${LIBRARIES}
)

#include(CTest)
#include(Catch)
catch_discover_tests(${TEST_EXE})

# set(OUTPUT_DIRECTORY ${ROOT_DIRECTORY}/Bin/)
# set_target_properties(${TEST_EXE} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
# set_target_properties(${TEST_EXE} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
# set_target_properties(${TEST_EXE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})

install(TARGETS ${TEST_EXE}
    ARCHIVE DESTINATION Bin
    LIBRARY DESTINATION Bin
    RUNTIME DESTINATION Bin)