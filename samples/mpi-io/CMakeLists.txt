## mpitest
add_executable(mpi-io main.c)
set_property(TARGET mpi-io PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mpi-io astaroth_core astaroth_utils)
