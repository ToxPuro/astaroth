## mpitest
add_executable(kernel-fusion-test main.cc)
set_property(TARGET kernel-fusion-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(kernel-fusion-test astaroth_core astaroth_utils)
