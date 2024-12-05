## Astaroth Kernels
if (PACKED_DATA_TRANSFERS)
    add_definitions(-DPACKED_DATA_TRANSFERS=1)
else ()
    add_definitions(-DPACKED_DATA_TRANSFERS=0)
endif ()

if (RUNTIME_COMPILATION)
	add_library(kernels empty.cu)
else()
	add_library(kernels kernels.cu logging.c)
endif()

target_link_libraries(kernels PUBLIC acc-runtime-headers)
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (USE_HIP)
    set_source_files_properties(kernels.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties(empty.cu PROPERTIES LANGUAGE CXX)
    set_property(TARGET kernels PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
else()
    set_property(TARGET kernels PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
