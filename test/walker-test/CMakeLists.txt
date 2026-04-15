## mpitest
add_subdirectory(matplotlib-cpp)
add_executable(walker-test main.cc)
target_link_libraries(walker-test astaroth_core astaroth_utils matplotlib_cpp)
