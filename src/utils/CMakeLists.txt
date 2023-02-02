## Astaroth Utils
add_library(modelsolver modelsolver.cc)
target_link_libraries(modelsolver astaroth_core)

add_library(astaroth_utils config_loader.c memory.c verification.c modelreduce.c logging.c)
target_link_libraries(astaroth_utils modelsolver astaroth_core)

target_compile_options(astaroth_utils PRIVATE "-mavx")
