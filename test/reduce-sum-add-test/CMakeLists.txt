## mpitest
add_executable(reduce-sum-add-test main.cc)
set_property(TARGET reduce-sum-add-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(reduce-sum-add-test astaroth_core astaroth_utils)
