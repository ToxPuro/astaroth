## mpitest
add_executable(inplace_gaussian-test main.cc)
set_property(TARGET inplace_gaussian-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(inplace_gaussian-test astaroth_core astaroth_utils)
