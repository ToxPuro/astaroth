add_executable(devicetest main.cc)
set_property(TARGET devicetest PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(devicetest astaroth_core astaroth_utils)
