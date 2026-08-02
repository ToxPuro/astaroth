## mpitest
add_executable(integration-test main.cc)
target_link_libraries(integration-test astaroth_core astaroth_utils)
find_package(GSL)
if(TARGET GSL::gsl)
  target_link_libraries(integration-test astaroth_core astaroth_utils GSL::gsl)
endif()
