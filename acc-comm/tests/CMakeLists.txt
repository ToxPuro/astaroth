## Packages
find_package(MPI REQUIRED COMPONENTS C)

## Automated tests
add_executable(unit-test unit_test.cc)
target_link_libraries(unit-test PRIVATE acm)
# add_test(NAME UnitTest COMMAND unit-test)

add_executable(c-test c_test.c)
target_link_libraries(c-test PRIVATE MPI::MPI_C acm)
add_test(NAME C_Test COMMAND c-test)

add_executable(comm-test comm_test.cc)
target_link_libraries(comm-test PRIVATE MPI::MPI_C acm)
add_test(NAME Comm_Test COMMAND comm-test)



