## cpptest
add_executable(cpptest main.cc)

set_property(TARGET cpptest PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cpptest PRIVATE astaroth_core astaroth_utils)
