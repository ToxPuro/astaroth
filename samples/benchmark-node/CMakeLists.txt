## benchmark
add_executable(benchmark-node main.c)
set_property(TARGET benchmark-node PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(benchmark-node astaroth_core astaroth_utils)
