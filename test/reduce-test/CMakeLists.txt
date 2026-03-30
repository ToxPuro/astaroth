## mpitest
add_executable(reduce-test main.cc)
set_property(TARGET reduce-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(reduce-test astaroth_core astaroth_utils)
