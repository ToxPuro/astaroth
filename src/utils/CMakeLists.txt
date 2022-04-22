## Astaroth Utils
if (BUILD_SHARED)
  add_library(astaroth_utils SHARED config_loader.c memory.c verification.c modelsolver.c modelreduce.c)
else()
  add_library(astaroth_utils STATIC config_loader.c memory.c verification.c modelsolver.c modelreduce.c)
endif()

target_link_libraries(astaroth_utils astaroth_core)

target_compile_options(astaroth_utils PRIVATE "-mavx")
