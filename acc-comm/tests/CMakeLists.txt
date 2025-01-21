## Packages

## Automated tests
add_executable(unit-test unit_test.cc)
target_link_libraries(unit-test PRIVATE acm)
# add_test(NAME UnitTest COMMAND unit-test)

add_executable(c-test c_test.c)
target_link_libraries(c-test PRIVATE acm)
add_test(NAME C_Test COMMAND c-test)

add_executable(comm-test comm_test.cc)
target_link_libraries(comm-test PRIVATE acm)
add_test(NAME Comm_Test COMMAND comm-test)

set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
find_package(MPI COMPONENTS C)
if (MPI_C_FOUND)
    add_executable(acm-mpi-test mpi-test.cc)
    target_link_libraries(acm-mpi-test PRIVATE MPI::MPI_C acm)
    add_test(NAME MPI_Test COMMAND acm-mpi-test)
else()
    message(WARNING "MPI not found, skipping acm MPI tests")
endif()
