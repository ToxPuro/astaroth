## Astaroth Utils
#target_compile_options(modelsolver PRIVATE "-mavx")

if(RUNTIME_COMPILATION)
	add_library(astaroth_utils empty.c)
else()
	add_library(modelsolver modelsolver.cc) # Change to modelsolver-legacy-do-not-modify.c and enable -mavx below to use the pure-C model solution used since at least the Vaisala 2021 paper
	target_link_libraries(modelsolver astaroth_core)
	add_library(astaroth_utils config_loader.cpp memory.c verification.c modelreduce.c)
	target_link_libraries(astaroth_utils astaroth_core)
endif()

