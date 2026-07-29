## mpitest
add_executable(grid-calculation main.cc)
find_package(GSL REQUIRED)
target_link_libraries(grid-calculation astaroth_core astaroth_utils GSL::gsl)
