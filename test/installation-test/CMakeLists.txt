## mpitest
cmake_minimum_required(VERSION 3.20)

project(SOR_STANDALONE)

find_package(Astaroth REQUIRED)
add_executable(sor-test main.cc)
target_link_libraries(sor-test PRIVATE Astaroth::astaroth_core Astaroth::astaroth_utils)
