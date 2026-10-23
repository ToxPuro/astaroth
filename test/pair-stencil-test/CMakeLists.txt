## pair-stencil-test
add_executable(pair-stencil-test main.cc)
set_property(TARGET pair-stencil-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(pair-stencil-test astaroth_core astaroth_utils)
