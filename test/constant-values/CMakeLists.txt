## mpitest
add_executable(constant-values main.cc)
set_property(TARGET constant-values PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(constant-values astaroth_core astaroth_utils)
