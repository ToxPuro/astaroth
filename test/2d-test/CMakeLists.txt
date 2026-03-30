## mpitest
add_executable(2d-test main.cc)
set_property(TARGET 2d-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(2d-test astaroth_core astaroth_utils)
