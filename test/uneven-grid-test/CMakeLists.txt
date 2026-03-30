## mpitest
add_executable(uneven-grid-test main.cc)
set_property(TARGET uneven-grid-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(uneven-grid-test astaroth_core astaroth_utils)
