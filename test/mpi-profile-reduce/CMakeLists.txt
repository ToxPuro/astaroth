## mpitest
add_executable(mpi-profile-reduce main.cc)
set_property(TARGET mpi-profile-reduce PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mpi-profile-reduce astaroth_core astaroth_utils)
