## mpitest
add_executable(integration-test main.cc)
find_package(GSL REQUIRED)
target_link_libraries(integration-test astaroth_core astaroth_utils GSL::gsl GSL::gslcblas)
