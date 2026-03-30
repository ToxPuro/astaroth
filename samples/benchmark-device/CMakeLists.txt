## benchmark
add_executable(benchmark-device main.cc)
set_property(TARGET benchmark-device PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(benchmark-device astaroth_core astaroth_utils)
