## Astaroth Utils
#target_compile_options(modelsolver PRIVATE "-mavx")
option(OPENMP_ENABLED "Enables OPENMP (used in modelsolver for now)" ON)

add_library(astaroth_utils)
target_sources(
	astaroth_utils
	PRIVATE
	config_loader.cc
	utils.cc
)
set_target_properties(
	astaroth_utils
	PROPERTIES
		POSITION_INDEPENDENT_CODE ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(
	astaroth_utils
	PUBLIC
	astaroth_core
)
ac_link_mpi(astaroth_utils)

if(NOT RUNTIME_COMPILATION)
	target_sources(
		astaroth_utils
		PRIVATE
		memory.cc
		modelreduce.cc
		modelsolver.cc
		profiles.cc
		verification.cc
	)

	add_library(astaroth_random astaroth_random.cc)
	set_property(TARGET astaroth_random PROPERTY POSITION_INDEPENDENT_CODE ON)
	target_link_libraries(astaroth_random astaroth_core)

	add_library(astaroth_forcing astaroth_forcing.cc helical_forcing.cc)
	set_property(TARGET astaroth_forcing PROPERTY POSITION_INDEPENDENT_CODE ON)
	target_link_libraries(astaroth_forcing astaroth_core astaroth_random)
	#TP: for parallelizing modelsolver
	if(OPENMP_ENABLED)
		find_package(OpenMP)
		if (OPENMP_FOUND)
			target_link_libraries(astaroth_utils PRIVATE OpenMP::OpenMP_C)
		endif()
	endif()
endif()

install(TARGETS
    astaroth_utils
    EXPORT AstarothTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
