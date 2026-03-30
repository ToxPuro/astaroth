## mpitest
add_executable(boundary-halo-exchange-test main.cc)
set_property(TARGET boundary-halo-exchange-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(boundary-halo-exchange-test astaroth_core astaroth_utils)
