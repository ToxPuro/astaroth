## mpitest
add_executable(use-scalar-reduce main.cc)
set_property(TARGET use-scalar-reduce PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(use-scalar-reduce astaroth_core astaroth_utils)
