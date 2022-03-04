## Astaroth Utils
add_library(astaroth_utils STATIC config_loader.c memory.c verification.c modelsolver.c modelreduce.c)
target_link_libraries(astaroth_utils astaroth_core)

target_compile_options(astaroth_utils PRIVATE "-mavx")
