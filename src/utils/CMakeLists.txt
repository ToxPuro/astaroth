## Astaroth Utils
add_library(astaroth_utils config_loader.c memory.c verification.c modelsolver.c modelreduce.c logging.c)
target_link_libraries(astaroth_utils astaroth_core)

target_compile_options(astaroth_utils PRIVATE "-mavx")
