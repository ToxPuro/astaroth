option(USE_HEFFTE "Whether to use HeFFTe for FFT" OFF)
## Astaroth Kernels
if (PACKED_DATA_TRANSFERS)
    add_definitions(-DPACKED_DATA_TRANSFERS=1)
else ()
    add_definitions(-DPACKED_DATA_TRANSFERS=0)
endif ()

if (USE_HIP)
	set_source_files_properties(
		kernels.cu
		reductions.cu
		reindex.cu
		transpose.cu
		PROPERTIES
			LANGUAGE HIP
	)
endif()
 
if (CPU_BUILD)
	set_source_files_properties(
		kernels.cu
		reductions.cu
		reindex.cu
		PROPERTIES
			LANGUAGE CXX
	)
endif()

add_library(
	astaroth_kernels
	OBJECT
	$<TARGET_OBJECTS:ac_cuda_wrappers>
	$<TARGET_OBJECTS:astaroth_helpers>
	reindex.cu
)

set_target_properties(
	astaroth_kernels
	PROPERTIES
	POSITION_INDEPENDENT_CODE ON
)
target_needs_separable_compilation(astaroth_kernels)
target_needs_device_side_linking(astaroth_kernels)

if (NOT CPU_BUILD)
	if (USE_HIP)
		set_target_properties(
			astaroth_kernels
			PROPERTIES
			HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
		)
	else()
		set_target_properties(
			astaroth_kernels
			PROPERTIES
			CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
		)
	endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	target_compile_definitions(astaroth_kernels PRIVATE -DACC_ARRAY_BOUND_CHECKS=1)
else()
	target_compile_definitions(astaroth_kernels PRIVATE -DACC_ARRAY_BOUND_CHECKS=0)
endif()

target_include_directories(
	astaroth_kernels
	PRIVATE
	${AC_BASE_PATH}/acc-runtime/api
	${AC_BASE_PATH}/include
	${AC_BASE_PATH}/src/core/kernels
	${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(astaroth_kernels PRIVATE ac_cuda_wrappers acc-runtime-headers)
ac_link_mpi(astaroth_kernels)

if(CPU_BUILD)
	target_sources(astaroth_kernels PRIVATE transpose.cc)
else()
	target_sources(astaroth_kernels PRIVATE transpose.cu)
endif()

if (RUNTIME_COMPILATION)
	target_link_libraries(astaroth_kernels PRIVATE acc-runtime-driver)
else()
	target_sources(
		astaroth_kernels
		PRIVATE
		kernels.cu
		reductions.cu
	)

	if (NOT USE_HIP AND CPU_BUILD)
		target_compile_options(
			astaroth_kernels
			PRIVATE
			$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
		)
        endif()

	if(FFT_ENABLED)
		if(CPU_BUILD)
			target_sources(
				astaroth_kernels
				PRIVATE
				kissfft/kiss_fft.c
				kissfft/kiss_fftnd.c
				cpufft.cc
			)
			if(DOUBLE_PRECISION)
				target_compile_definitions(astaroth_kernels PUBLIC kiss_fft_scalar=double)
			endif()
		else()
			if(USE_HIP)
				find_package(rocfft REQUIRED)
				target_link_libraries(astaroth_kernels PUBLIC roc::rocfft)
			else()
				target_link_libraries(astaroth_kernels PUBLIC CUDA::cufft)
			endif()

			if(USE_HEFFTE)
				target_sources(astaroth_kernels PRIVATE heffte.cc)
				if(DEFINED ENV{HEFFTE_DIR})
					target_include_directories(
						astaroth_kernels
						PRIVATE
						"$ENV{HEFFTE_DIR}/include"
						"$ENV{HEFFTE_DIR}/build/include"
					)
					target_link_directories(
						astaroth_kernels
						PUBLIC
						"$ENV{HEFFTE_DIR}/build"
					)
				endif()
				target_link_libraries(astaroth_kernels PUBLIC heffte)
			else()
				if(USE_HIP)
					target_sources(astaroth_kernels PRIVATE rocfft.cc)
				        #TP: there could a cleaner way to do this put to enable debuging now on LUMI explicitly include
				        target_include_directories(astaroth_kernels PUBLIC /opt/rocm/include/rocfft)
				else()
					target_sources(astaroth_kernels PRIVATE cufft.cc)
				endif()
			endif()
		endif()
	else()
		target_sources(astaroth_kernels PRIVATE nofft.cc)
	endif()
endif()
