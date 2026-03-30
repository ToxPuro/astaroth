## mpitest
add_executable(planetest main.cc)
set_property(TARGET planetest PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(planetest astaroth_core astaroth_utils)
