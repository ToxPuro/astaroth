## mpitest
add_executable(mpi-io-multithreaded main.cc)
set_property(TARGET mpi-io-multithreaded PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mpi-io-multithreaded astaroth_core astaroth_utils)
