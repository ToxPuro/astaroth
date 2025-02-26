file(GLOB TEST_SOURCES "*.cc" "*.c" "detail/*.cc" "detail/*.c")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WLE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE acm)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
