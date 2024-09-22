## Astaroth Utils
#target_compile_options(modelsolver PRIVATE "-mavx")
option(BUILD_TESTS        "Builds tests in test subdirectory."            OFF)

if(RUNTIME_COMPILATION)
	add_library(astaroth_utils empty.c)
else()
	#TP: if the modelsolver is too slow can use OpenMP to parallelize it across cores
	#find_package(OpenMP)
	#if (OPENMP_FOUND)
	#	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	#       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	#	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	#endif()
	if(NOT BUILD_TESTS)
		add_library(astaroth_utils memory.c verification.c modelreduce.c modelsolver.cc)
	else()
		add_library(astaroth_utils memory.c verification.c modelreduce.c)
	endif()
	target_link_libraries(astaroth_utils astaroth_core)
endif()

