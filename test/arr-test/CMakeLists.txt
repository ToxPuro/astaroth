## mpitest
add_executable(arr-test main.cc)
set_property(TARGET arr-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(arr-test astaroth_core astaroth_utils)
