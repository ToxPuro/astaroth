## ctest
add_executable(ctest main.c)

set_property(TARGET ctest PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(ctest PRIVATE astaroth_core astaroth_utils)
