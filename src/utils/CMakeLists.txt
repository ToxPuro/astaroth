## Astaroth Utils
#target_compile_options(modelsolver PRIVATE "-mavx")
option(OPENMP_ENABLED "Enables OPENMP (used in modelsolver for now)" ON)

if(RUNTIME_COMPILATION)
	add_library(astaroth_utils config_loader.cc)
	target_link_libraries(astaroth_utils astaroth_core)
else()

	add_library(astaroth_utils memory.cc verification.cc modelreduce.cc modelsolver.cc profiles.cc config_loader.cc)
	target_link_libraries(astaroth_utils PUBLIC astaroth_core)

	add_library(astaroth_random astaroth_random.cc)
	target_link_libraries(astaroth_random astaroth_core)

	add_library(astaroth_forcing astaroth_forcing.cc)
	target_link_libraries(astaroth_forcing astaroth_core astaroth_random)
	#TP: for parallelizing modelsolver
	if(OPENMP_ENABLED)
		find_package(OpenMP)
		if (OPENMP_FOUND)
			target_link_libraries(astaroth_utils PRIVATE OpenMP::OpenMP_C)
		endif()
	endif()
endif()

