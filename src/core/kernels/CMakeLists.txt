## Astaroth Kernels
if (PACKED_DATA_TRANSFERS)
    add_definitions(-DPACKED_DATA_TRANSFERS=1)
else ()
    add_definitions(-DPACKED_DATA_TRANSFERS=0)
endif ()

if (USE_HIP)
    set_source_files_properties(kernels.cu transpose.cu reductions.cu reindex.cu PROPERTIES LANGUAGE HIP)
endif()
 
if (CPU_BUILD)
	set_source_files_properties(kernels.cu transpose.cu reductions.cu reindex.cu PROPERTIES LANGUAGE CXX)
endif()

if (RUNTIME_COMPILATION)
	add_library(static_kernels STATIC empty.cc)
else()
	if(FFT_ENABLED)
		if(USE_HIP)
			add_library(ac_fft STATIC rocfft.cc)
           	        find_package(rocfft REQUIRED)
           	        target_link_libraries(ac_fft PUBLIC roc::rocfft)
        	        #TP: there could a cleaner way to do this put to enable debuging now on LUMI explicitly include
        	        target_include_directories(ac_fft PUBLIC /opt/rocm/include/rocfft)
		else()
			add_library(ac_fft STATIC cufft.cc)
            		target_link_libraries(ac_fft PUBLIC CUDA::cufft)
		endif()
	else()
		add_library(ac_fft STATIC nofft.cc)
	endif()

	target_include_directories(ac_fft PRIVATE ${AC_BASE_PATH}/acc-runtime/api ${AC_BASE_PATH}/include)
	target_link_libraries(ac_fft PRIVATE ac_cuda_wrappers)

	add_library(ac_transpose STATIC transpose.cu)
	target_include_directories(ac_transpose PRIVATE ${AC_BASE_PATH}/acc-runtime/api ${AC_BASE_PATH}/include)
	target_link_libraries(ac_transpose PRIVATE ac_cuda_wrappers)

	add_library(ac_reindex STATIC reindex.cu)
	target_include_directories(ac_reindex PRIVATE ${AC_BASE_PATH}/acc-runtime/api ${AC_BASE_PATH}/include)
	target_link_libraries(ac_reindex PRIVATE ac_cuda_wrappers)

	add_library(static_kernels STATIC kernels.cu reductions.cu)
	target_link_libraries(static_kernels PUBLIC ac_fft ac_transpose ac_reindex)
endif()

target_link_libraries(static_kernels PRIVATE ac_cuda_wrappers astaroth_helpers)
target_link_libraries(static_kernels PUBLIC acc-runtime-headers acc-runtime-driver)
target_include_directories(static_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (USE_HIP)
    set_property(TARGET static_kernels PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
else()
    set_property(TARGET static_kernels PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	target_compile_definitions(static_kernels PRIVATE -DACC_ARRAY_BOUND_CHECKS=1)
else()
	target_compile_definitions(static_kernels PRIVATE -DACC_ARRAY_BOUND_CHECKS=0)
endif()

add_library(kernels empty.cc)
target_link_libraries(kernels PRIVATE static_kernels)
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)
