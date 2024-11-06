## Astaroth Utils
#target_compile_options(modelsolver PRIVATE "-mavx")
option(BUILD_TESTS        "Builds tests in test subdirectory."            OFF)
option(BUILD_MODEL        "Builds model solutions to test against lib implementations."  OFF)

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
	if(BUILD_MODEL)
		add_library(astaroth_utils memory.cc verification.cc modelreduce.cc modelsolver.cc profiles.cc)
	else()
		add_library(astaroth_utils memory.cc verification.cc  profiles.cc)
	endif()
	target_link_libraries(astaroth_utils astaroth_core)

	add_library(astaroth_random astaroth_random.cc)
	target_link_libraries(astaroth_random astaroth_core)

	add_library(astaroth_forcing astaroth_forcing.cc)
	target_link_libraries(astaroth_forcing astaroth_core astaroth_random)
endif()

