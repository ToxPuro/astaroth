## mpitest
add_executable(average-stencil main.cc)
set_property(TARGET average-stencil PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(average-stencil astaroth_core astaroth_utils)
