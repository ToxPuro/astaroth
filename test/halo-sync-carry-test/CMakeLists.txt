## halo-sync-carry-test
add_executable(halo-sync-carry-test main.cc)
set_property(TARGET halo-sync-carry-test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(halo-sync-carry-test astaroth_core astaroth_utils)
