add_executable(benchmark-thrust main.cu)
target_link_libraries(benchmark-thrust PRIVATE astaroth_core astaroth_utils)
