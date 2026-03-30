## mpitest
add_executable(field-arr-test main.cc)
set_property(TARGET field-arr-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(field-arr-test astaroth_core astaroth_utils)
