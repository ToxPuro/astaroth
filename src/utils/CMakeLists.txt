## Astaroth Utils
add_library(modelsolver modelsolver.cc) # Change to modelsolver-legacy-do-not-modify.c and enable -mavx below to use the pure-C model solution used since at least the Vaisala 2021 paper
target_link_libraries(modelsolver astaroth_core)
#target_compile_options(modelsolver PRIVATE "-mavx")

add_library(astaroth_random astaroth_random.cc)
target_link_libraries(astaroth_random astaroth_core)

add_library(astaroth_forcing astaroth_forcing.cc)
target_link_libraries(astaroth_forcing astaroth_core astaroth_random)

add_library(astaroth_utils config_loader.c memory.c verification.c modelreduce.c profiles.c logging.c)
target_link_libraries(astaroth_utils modelsolver astaroth_core)
target_include_directories(astaroth_utils PRIVATE ${CMAKE_SOURCE_DIR})

