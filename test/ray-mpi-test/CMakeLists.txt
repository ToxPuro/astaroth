## mpitest
add_executable(ray-mpi-test main.cc)
set_property(TARGET ray-mpi-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ray-mpi-test astaroth_core astaroth_utils)
