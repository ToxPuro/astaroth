## mpitest
add_executable(grid-calculation main.cc)
target_link_libraries(grid-calculation astaroth_core astaroth_utils)
find_package(GSL)
if(TARGET GSL::GSL)
  target_link_libraries(poisson-test astaroth_core astaroth_utils GSL::gsl)
endif()
