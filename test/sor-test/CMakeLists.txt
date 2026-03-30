## mpitest
add_executable(sor-test main.cc)
set_property(TARGET sor-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(sor-test astaroth_core astaroth_utils)
