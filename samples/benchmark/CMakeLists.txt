## benchmark
add_executable(benchmark main.cc)
set_property(TARGET benchmark PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(benchmark astaroth_core astaroth_utils)
