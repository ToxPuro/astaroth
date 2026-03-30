## mpitest
add_executable(uneven-sor-test main.cc)
set_property(TARGET uneven-sor-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(uneven-sor-test astaroth_core astaroth_utils)
