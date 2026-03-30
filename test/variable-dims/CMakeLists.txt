## mpitest
add_executable(variable-dims main.cc)
set_property(TARGET variable-dims PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(variable-dims astaroth_core astaroth_utils)
