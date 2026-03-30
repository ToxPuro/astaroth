## mpitest
add_executable(single-precision-test main.cc)
set_property(TARGET single-precision-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(single-precision-test astaroth_core astaroth_utils)
