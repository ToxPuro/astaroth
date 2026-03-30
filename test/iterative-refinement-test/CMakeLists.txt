## mpitest
add_executable(iterative-refinement-test main.cc)
set_property(TARGET iterative-refinement-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(iterative-refinement-test astaroth_core astaroth_utils)
