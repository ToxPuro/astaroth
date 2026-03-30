## mpitest
add_executable(small-grid-test main.cc)
set_property(TARGET small-grid-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(small-grid-test astaroth_core astaroth_utils)
