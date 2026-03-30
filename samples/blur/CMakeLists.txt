## benchmark
add_executable(blur main.c)
set_property(TARGET blur PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(blur astaroth_core astaroth_utils)
