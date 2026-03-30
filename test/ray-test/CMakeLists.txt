## mpitest
add_executable(ray-test main.cc)
set_property(TARGET ray-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ray-test astaroth_core astaroth_utils)
