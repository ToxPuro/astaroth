## Astaroth Utils
#target_compile_options(modelsolver PRIVATE "-mavx")
option(BUILD_TESTS        "Builds tests in test subdirectory."            OFF)

if(RUNTIME_COMPILATION)
	add_library(astaroth_utils empty.c)
else()
	add_library(astaroth_utils config_loader.cpp memory.c verification.c modelreduce.c)
if(NOT BUILD_TESTS)
	add_library(modelsolver modelsolver.cc) # Change to modelsolver-legacy-do-not-modify.c and enable -mavx below to use the pure-C model solution used since at least the Vaisala 2021 paper
	target_link_libraries(modelsolver astaroth_core)
	target_link_libraries(astaroth_utils astaroth_core modelsolver)
else()
	target_link_libraries(astaroth_utils astaroth_core)
endif()
endif()

