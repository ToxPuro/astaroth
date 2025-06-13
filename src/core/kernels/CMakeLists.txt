## Astaroth Kernels
if (PACKED_DATA_TRANSFERS)
    add_definitions(-DPACKED_DATA_TRANSFERS=1)
else ()
    add_definitions(-DPACKED_DATA_TRANSFERS=0)
endif ()

if (RUNTIME_COMPILATION)
	add_library(static_kernels STATIC logging.c)
else()
	add_library(static_kernels STATIC kernels.cu logging.c)
endif()

target_link_libraries(static_kernels PRIVATE astaroth_helpers)
target_link_libraries(static_kernels PUBLIC acc-runtime-headers)
target_include_directories(static_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (USE_HIP)
    set_source_files_properties(kernels.cu PROPERTIES LANGUAGE HIP)
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
