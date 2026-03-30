## mpitest
add_executable(bicgstab-test main.cc)
set_property(TARGET bicgstab-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(bicgstab-test astaroth_core astaroth_utils)
