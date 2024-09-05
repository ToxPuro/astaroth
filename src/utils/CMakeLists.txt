## Astaroth Utils
#target_compile_options(modelsolver PRIVATE "-mavx")
option(BUILD_TESTS        "Builds tests in test subdirectory."            OFF)

if(RUNTIME_COMPILATION)
	add_library(astaroth_utils empty.c)
else()
	if(NOT BUILD_TESTS)
		add_library(astaroth_utils memory.c verification.c modelreduce.c modelsolver.cc)
	else()
		add_library(astaroth_utils memory.c verification.c modelreduce.c)
	endif()
	target_link_libraries(astaroth_utils astaroth_core)
endif()

