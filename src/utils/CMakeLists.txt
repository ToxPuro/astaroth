## Astaroth Utils
add_library(astaroth_utils STATIC config_loader.c memory.c verification.c modelsolver.c)
add_dependencies(astaroth_utils dsl_headers)
